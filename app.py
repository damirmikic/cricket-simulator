import streamlit as st
import pandas as pd
import json
import os
import random
import time
from collections import defaultdict

# --- Core Data Processing Functions ---

def get_match_phase(over_num):
    """Determines the phase of the match based on the over number."""
    if over_num <= 5: return 'Powerplay'
    if over_num <= 14: return 'Middle'
    return 'Death'

def process_all_matches_and_players(uploaded_files):
    """
    Processes all uploaded files to extract innings, player, phase, and ball outcome stats.
    """
    all_innings_data = []
    player_batting_stats = defaultdict(lambda: defaultdict(int))
    player_bowling_stats = defaultdict(lambda: defaultdict(int))
    teams_players = defaultdict(set)
    team_phase_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    team_ball_outcomes = defaultdict(lambda: defaultdict(int))

    for f in uploaded_files:
        try:
            # Reset file pointer for each read
            f.seek(0)
            data = json.load(f)
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError) as e:
            st.warning(f"Could not process file {getattr(f, 'name', 'N/A')}: {e}")
            continue

        info = data.get('info', {})
        match_date = info.get('dates', [None])[0]
        for team, players in info.get('players', {}).items():
            teams_players[team].update(players)
        
        winner = info.get('outcome', {}).get('winner')

        for inning in data.get('innings', []):
            team = inning.get('team')
            is_win = 1 if team == winner else 0
            
            for over in inning.get('overs', []):
                phase = get_match_phase(over['over'])
                for delivery in over.get('deliveries', []):
                    batter, bowler, runs = delivery['batter'], delivery['bowler'], delivery['runs']
                    is_legal = 'wides' not in delivery.get('extras', {}) and 'noballs' not in delivery.get('extras', {})
                    
                    # Phase & Outcome Stats
                    team_phase_stats[team][phase]['runs'] += runs['total']
                    team_ball_outcomes[team]['total_balls'] += 1
                    if is_legal:
                        team_phase_stats[team][phase]['balls'] += 1
                    if 'wickets' in delivery:
                        team_phase_stats[team][phase]['wickets'] += 1
                        team_ball_outcomes[team]['WICKET'] += 1
                    else:
                        team_ball_outcomes[team][runs['batter']] += 1
                    
                    # Player Stats
                    player_batting_stats[batter]['runs'] += runs['batter']
                    if is_legal: player_batting_stats[batter]['balls_faced'] += 1
                    player_bowling_stats[bowler]['runs_conceded'] += runs['total']
                    if is_legal: player_bowling_stats[bowler]['balls_bowled'] += 1
                    if 'wickets' in delivery and delivery['wickets'][0]['kind'] not in ['run out', 'retired hurt']:
                        player_bowling_stats[bowler]['wickets'] += 1

            # Aggregate innings data after processing all overs for it
            innings_runs = sum(team_phase_stats[team][p]['runs'] for p in team_phase_stats[team])
            innings_balls = sum(team_phase_stats[team][p]['balls'] for p in team_phase_stats[team])
            innings_wickets = sum(team_phase_stats[team][p]['wickets'] for p in team_phase_stats[team])
            all_innings_data.append({
                'match_date': match_date, 'team': team, 'total_runs': innings_runs,
                'wickets_lost': innings_wickets, 'overs_bowled': round(innings_balls / 6.0, 2),
                'run_rate': round((innings_runs * 6 / innings_balls) if innings_balls > 0 else 0, 2),
                'wicket_rate': round((innings_wickets * 6 / innings_balls) if innings_balls > 0 else 0, 2), # Per over
                'is_win': is_win
            })

    # Create DataFrames
    innings_df = pd.DataFrame(all_innings_data) if all_innings_data else pd.DataFrame()
    batting_df = pd.DataFrame.from_dict(player_batting_stats, orient='index')
    bowling_df = pd.DataFrame.from_dict(player_bowling_stats, orient='index')
    
    # Calculate advanced stats
    if not batting_df.empty:
        batting_df['strike_rate'] = round(batting_df['runs'] * 100 / batting_df['balls_faced'], 2)
    if not bowling_df.empty:
        bowling_df['economy'] = round(bowling_df['runs_conceded'] * 6 / bowling_df['balls_bowled'], 2)
        bowling_df['avg'] = round(bowling_df['runs_conceded'] / bowling_df['wickets'], 2).fillna(0)

    teams_players = {team: sorted(list(players)) for team, players in teams_players.items()}
    return innings_df, batting_df, bowling_df, teams_players, team_phase_stats, team_ball_outcomes

def calculate_form_stats(team, df):
    team_matches = df[df['team'] == team].sort_values('match_date', ascending=False)
    stats = {}
    for n in [5, 10]:
        last_n = team_matches.head(n)
        if not last_n.empty:
            stats[f'rr_L{n}'] = last_n['run_rate'].mean()
            stats[f'wr_L{n}'] = last_n['wicket_rate'].mean()
            if n == 5:
                stats['form_L5'] = ''.join(['W' if r['is_win'] else 'L' for _, r in last_n.iterrows()])
    return pd.Series(stats)

# --- Simulation Functions ---
def simulate_ball_by_phase(phase, phase_stats):
    """Simulates a single ball based on team's phase performance."""
    phase_data = phase_stats.get(phase, {})
    balls = phase_data.get('balls', 1) # Avoid division by zero
    rr = (phase_data.get('runs', 0) * 6) / balls if balls > 0 else 6.0
    wr = (phase_data.get('wickets', 0) * 6) / balls if balls > 0 else 0.5

    prob_wicket = wr / 6.0
    prob_dot = max(0.1, 0.5 - (rr / 20.0))
    prob_four = max(0.05, (rr / 60.0))
    prob_six = max(0.02, (rr / 90.0))
    
    outcomes = ['WICKET', 0, 1, 2, 4, 6]
    weights = [prob_wicket, prob_dot, 0.35, 0.05, prob_four, prob_six]
    norm_weights = [w / sum(weights) for w in weights]
    return random.choices(outcomes, norm_weights)[0]

# --- Streamlit UI ---
st.set_page_config(page_title="Cricket Predictive Model", layout="wide")
st.title("🏏 Cricket Predictive Model")
st.markdown("By **Virat** for **bet365**")

# Sidebar and Data Loading
st.sidebar.header("Upload Match Data")
uploaded_files = st.sidebar.file_uploader("Upload JSON match logs", type=['json'], accept_multiple_files=True)
if not uploaded_files and os.path.exists('335982.json'):
    st.sidebar.info("Showing data from the default example file.")
    with open('335982.json', 'rb') as f:
        from io import BytesIO
        bytes_io_wrapper = BytesIO(f.read())
        bytes_io_wrapper.name = '335982.json'
        uploaded_files = [bytes_io_wrapper]

if uploaded_files:
    data_load_state = st.text('Loading and processing data...')
    innings_df, batting_df, bowling_df, teams_players, team_phase_stats, team_ball_outcomes = process_all_matches_and_players(uploaded_files)
    data_load_state.text('Processing complete!')
    
    if not innings_df.empty:
        team_stats_form = innings_df.groupby('team').apply(lambda x: calculate_form_stats(x.name, innings_df)).round(2)
        team_stats = innings_df.groupby('team').agg(run_rate=('run_rate', 'mean'), wicket_rate=('wicket_rate', 'mean')).round(2)
        team_stats = team_stats.join(team_stats_form).reset_index()

        tab_titles = ["📊 Data Analysis", "👨‍💻 Player Performance", "📋 Lineup Creator", "⚡ Fast Simulator", "🆚 Advanced Simulator"]
        tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

        with tab1:
            st.header("Team Performance & Form Summary")
            st.dataframe(team_stats)
            
            st.header("Match Phase Analysis")
            team_to_analyze = st.selectbox("Select Team for Phase Analysis", team_stats['team'])
            if team_to_analyze:
                phase_data = []
                for phase in ['Powerplay', 'Middle', 'Death']:
                    stats = team_phase_stats[team_to_analyze][phase]
                    balls = stats.get('balls', 0)
                    runs = stats.get('runs', 0)
                    wickets = stats.get('wickets', 0)
                    phase_data.append({
                        'Phase': phase,
                        'Runs': runs, 'Balls': balls, 'Wickets': wickets,
                        'Run Rate': round((runs * 6 / balls) if balls else 0, 2),
                        'Wicket Rate': round((wickets * 6 / balls) if balls else 0, 2)
                    })
                st.dataframe(pd.DataFrame(phase_data))

            st.header("Ball Outcome DNA")
            team_to_dna = st.selectbox("Select Team for Ball Outcome Analysis", team_stats['team'])
            if team_to_dna:
                outcomes = team_ball_outcomes[team_to_dna]
                total = outcomes.get('total_balls', 1)
                outcome_data = {
                    "Outcome": ["Wickets", "Dots", "1s", "2s", "3s", "4s", "6s"],
                    "Percentage": [
                        outcomes.get('WICKET', 0) * 100 / total, outcomes.get(0, 0) * 100 / total,
                        outcomes.get(1, 0) * 100 / total, outcomes.get(2, 0) * 100 / total,
                        outcomes.get(3, 0) * 100 / total, outcomes.get(4, 0) * 100 / total,
                        outcomes.get(6, 0) * 100 / total,
                    ]
                }
                st.dataframe(pd.DataFrame(outcome_data), hide_index=True)


        with tab2:
            st.header("Player Performance")
            st.subheader("Batting Stats")
            st.dataframe(batting_df.sort_values('runs', ascending=False))
            st.subheader("Bowling Stats")
            st.dataframe(bowling_df.sort_values('wickets', ascending=False))

        with tab3:
            st.header("Lineup Creator")
            team_to_build = st.selectbox("Select Team to Build Lineup", teams_players.keys())
            if team_to_build:
                available_players = teams_players[team_to_build]
                st.multiselect(f"Select 11 Players for {team_to_build}", options=available_players, max_selections=11)

        with tab4:
            st.header("⚡ Fast Match Simulator (Team Averages)")
            num_sims = st.number_input("How many simulations?", min_value=1, max_value=10000, value=100)
            if st.button("▶️ Run Fast Simulation", use_container_width=True):
                # Using placeholder names and stats for brevity
                t1_name, t1_rr, t1_wr = "Team A", 8.5, 0.65
                t2_name, t2_rr, t2_wr = "Team B", 8.2, 0.70
                wins = {t1_name: 0, t2_name: 0}
                progress_bar = st.progress(0, text="Simulating...")
                for i in range(num_sims):
                    score1, _ = simulate_fast_innings(t1_rr, t1_wr)
                    score2, _ = simulate_fast_innings(t2_rr, t2_wr)
                    wins[t1_name if score1 > score2 else t2_name] += 1
                    progress_bar.progress((i + 1) / num_sims, text=f"Simulating... {i+1}/{num_sims}")
                
                win_p1 = (wins[t1_name] / num_sims) * 100
                st.metric(f"{t1_name} Win Probability", f"{win_p1:.1f}%")
                st.metric(f"{t2_name} Win Probability", f"{100-win_p1:.1f}%")

        with tab5:
            st.header("🆚 Advanced Simulator (Phase-Based)")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                team1_name = st.selectbox("Select Team 1", teams_players.keys(), key="p_team1")
            with pcol2:
                team2_name = st.selectbox("Select Team 2", teams_players.keys(), index=1 if len(teams_players.keys()) > 1 else 0, key="p_team2")

            if st.button("Run Advanced Simulation", use_container_width=True):
                st.subheader(f"Simulating: {team1_name} vs {team2_name}")
                
                # Innings 1
                score1, wickets1 = 0, 0
                for ball in range(120):
                    if wickets1 >= 10: break
                    phase = get_match_phase(ball // 6)
                    outcome = simulate_ball_by_phase(phase, team_phase_stats[team1_name])
                    if isinstance(outcome, int): score1 += outcome
                    else: wickets1 += 1
                
                # Innings 2
                score2, wickets2 = 0, 0
                target = score1 + 1
                for ball in range(120):
                    if wickets2 >= 10 or score2 >= target: break
                    phase = get_match_phase(ball // 6)
                    outcome = simulate_ball_by_phase(phase, team_phase_stats[team2_name])
                    if isinstance(outcome, int): score2 += outcome
                    else: wickets2 += 1
                
                st.header("Match Result")
                rcol1, rcol2 = st.columns(2)
                rcol1.metric(f"{team1_name} Score", f"{score1}/{wickets1}")
                rcol2.metric(f"{team2_name} Score", f"{score2}/{wickets2}")
                if score2 >= target:
                    st.subheader(f"🎉 {team2_name} wins!")
                else:
                    st.subheader(f"🎉 {team1_name} wins!")

    else:
        st.warning("Could not process uploaded files. Please check file format and content.")
else:
    st.info("👋 Welcome! Upload JSON match logs to begin.")

