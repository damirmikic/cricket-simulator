import streamlit as st
import pandas as pd
import json
import os
import random
import time
from collections import defaultdict
from io import BytesIO

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
                    
                    team_phase_stats[team][phase]['runs'] += runs['total']
                    team_ball_outcomes[team]['total_balls'] += 1
                    if is_legal:
                        team_phase_stats[team][phase]['balls'] += 1
                    if 'wickets' in delivery:
                        team_phase_stats[team][phase]['wickets'] += 1
                        team_ball_outcomes[team]['WICKET'] += 1
                    else:
                        team_ball_outcomes[team][runs['batter']] += 1
                    
                    player_batting_stats[batter]['runs'] += runs['batter']
                    if is_legal: player_batting_stats[batter]['balls_faced'] += 1
                    player_bowling_stats[bowler]['runs_conceded'] += runs['total']
                    if is_legal: player_bowling_stats[bowler]['balls_bowled'] += 1
                    if 'wickets' in delivery and delivery['wickets'][0]['kind'] not in ['run out', 'retired hurt']:
                        player_bowling_stats[bowler]['wickets'] += 1

            innings_runs = sum(team_phase_stats[team][p]['runs'] for p in team_phase_stats[team])
            innings_balls = sum(team_phase_stats[team][p]['balls'] for p in team_phase_stats[team])
            innings_wickets = sum(team_phase_stats[team][p]['wickets'] for p in team_phase_stats[team])
            all_innings_data.append({
                'match_date': match_date, 'team': team, 'total_runs': innings_runs,
                'wickets_lost': innings_wickets, 'overs_bowled': round(innings_balls / 6.0, 2),
                'run_rate': round((innings_runs * 6 / innings_balls) if innings_balls > 0 else 0, 2),
                'wicket_rate': round((innings_wickets * 6 / innings_balls) if innings_balls > 0 else 0, 2),
                'is_win': is_win
            })

    innings_df = pd.DataFrame(all_innings_data) if all_innings_data else pd.DataFrame()
    batting_df = pd.DataFrame.from_dict(player_batting_stats, orient='index')
    bowling_df = pd.DataFrame.from_dict(player_bowling_stats, orient='index')
    
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

def get_last_match_lineup(team, uploaded_files):
    last_date = None
    lineup = []
    for f in uploaded_files:
        try:
            f.seek(0)
            data = json.load(f)
            info = data.get('info', {})
            match_date = info.get('dates', [None])[0]
            if team in info.get('players', {}):
                if not last_date or (match_date and match_date > last_date):
                    last_date = match_date
                    lineup = info['players'][team]
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
            continue
    return lineup

# --- Simulation Functions ---
def simulate_fast_innings(team_stats, phase_stats):
    score, wickets, fours, sixes = 0, 0, 0, 0
    runs_at_1st_wicket = -1
    dismissal_methods = []

    rr = team_stats['run_rate']
    wr = team_stats['wicket_rate']
    
    for _ in range(120):
        if wickets >= 10: break

        prob_wicket = wr / 6.0
        prob_dot = max(0.1, 0.5 - (rr / 20.0))
        prob_four = max(0.05, (rr / 60.0))
        prob_six = max(0.02, (rr / 90.0))
        
        outcomes = ['WICKET', 0, 1, 2, 4, 6]
        weights = [prob_wicket, prob_dot, 0.35, 0.05, prob_four, prob_six]
        norm_weights = [w / sum(weights) for w in weights]
        outcome = random.choices(outcomes, norm_weights)[0]
        
        if outcome == 'WICKET':
            wickets += 1
            if runs_at_1st_wicket == -1: runs_at_1st_wicket = score
            dismissal_methods.append(random.choice(['Caught', 'Bowled', 'LBW', 'Stumped', 'Run Out']))
        else:
            score += outcome
            if outcome == 4: fours += 1
            if outcome == 6: sixes += 1
            
    return {
        "score": score, "wickets": wickets, "fours": fours, "sixes": sixes,
        "boundaries": fours + sixes, "runs_at_1st_wicket": runs_at_1st_wicket if runs_at_1st_wicket != -1 else score,
        "dismissal_methods": dismissal_methods
    }

def simulate_ball_by_phase(phase, phase_stats):
    phase_data = phase_stats.get(phase, {})
    balls = phase_data.get('balls', 1)
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

st.sidebar.header("Upload Match Data")
uploaded_files = st.sidebar.file_uploader("Upload JSON match logs", type=['json'], accept_multiple_files=True)
if not uploaded_files and os.path.exists('335982.json'):
    st.sidebar.info("Showing data from the default example file.")
    with open('335982.json', 'rb') as f:
        bytes_io_wrapper = BytesIO(f.read())
        bytes_io_wrapper.name = '335982.json'
        uploaded_files = [bytes_io_wrapper]

if uploaded_files:
    data_load_state = st.text('Loading and processing data...')
    innings_df, batting_df, bowling_df, teams_players, team_phase_stats, team_ball_outcomes = process_all_matches_and_players(uploaded_files)
    data_load_state.text('Processing complete!')
    
    if not innings_df.empty:
        team_stats_form = innings_df.groupby('team').apply(lambda x: calculate_form_stats(x.name, innings_df)).round(2)
        team_stats_agg = innings_df.groupby('team').agg(run_rate=('run_rate', 'mean'), wicket_rate=('wicket_rate', 'mean')).round(2)
        team_stats = team_stats_agg.join(team_stats_form).reset_index()

        tab_titles = ["📊 Data Analysis", "👨‍💻 Player Performance", "📋 Lineup Creator", "⚡ Fast Simulator", "🆚 Advanced Simulator"]
        tabs = st.tabs(tab_titles)

        with tabs[0]: # Data Analysis
            st.header("Team Performance & Form Summary")
            st.dataframe(team_stats)
            st.header("Match Phase Analysis")
            team_to_analyze = st.selectbox("Select Team for Phase Analysis", team_stats['team'])
            if team_to_analyze:
                phase_data = []
                for phase in ['Powerplay', 'Middle', 'Death']:
                    stats = team_phase_stats[team_to_analyze][phase]
                    balls, runs, wickets = stats.get('balls', 0), stats.get('runs', 0), stats.get('wickets', 0)
                    phase_data.append({
                        'Phase': phase, 'Runs': runs, 'Balls': balls, 'Wickets': wickets,
                        'Run Rate': round((runs * 6 / balls) if balls else 0, 2),
                        'Wicket Rate': round((wickets * 6 / balls) if balls else 0, 2)
                    })
                st.dataframe(pd.DataFrame(phase_data))

        with tabs[1]: # Player Performance
            st.header("Player Performance")
            st.subheader("Batting Stats")
            st.dataframe(batting_df.sort_values('runs', ascending=False))
            st.subheader("Bowling Stats")
            st.dataframe(bowling_df.sort_values('wickets', ascending=False))

        with tabs[2]: # Lineup Creator
            st.header("Lineup Creator")
            team_to_build = st.selectbox("Select Team to Build Lineup", teams_players.keys())
            if team_to_build:
                suggested_lineup = get_last_match_lineup(team_to_build, uploaded_files)
                # FIX: Ensure the default list does not exceed max_selections
                suggested_lineup = suggested_lineup[:11] 
                st.multiselect(
                    f"Select 11 Players for {team_to_build} (pre-filled with last match's lineup)",
                    options=teams_players[team_to_build],
                    default=suggested_lineup,
                    max_selections=11
                )

        with tabs[3]: # Fast Simulator
            st.header("⚡ Fast Match Simulator")
            sim_cols = st.columns(2)
            team_list = list(teams_players.keys())
            t1_name = sim_cols[0].selectbox("Select Team 1", team_list, key="sim_t1")
            t2_name = sim_cols[1].selectbox("Select Team 2", team_list, index=1 if len(team_list) > 1 else 0, key="sim_t2")

            st.subheader("Team Parameters")
            param_cols = st.columns(2)
            t1_stats = team_stats[team_stats['team'] == t1_name].iloc[0]
            t2_stats = team_stats[team_stats['team'] == t2_name].iloc[0]
            with param_cols[0]:
                st.metric(label=f"{t1_name} Avg. Run Rate", value=f"{t1_stats['run_rate']:.2f}")
                st.metric(label=f"{t1_name} Avg. Wicket Rate", value=f"{t1_stats['wicket_rate']:.2f}")
            with param_cols[1]:
                st.metric(label=f"{t2_name} Avg. Run Rate", value=f"{t2_stats['run_rate']:.2f}")
                st.metric(label=f"{t2_name} Avg. Wicket Rate", value=f"{t2_stats['wicket_rate']:.2f}")
            
            num_sims = st.number_input("How many simulations?", 1, 10000, 100)

            if st.button("▶️ Run Fast Simulation", use_container_width=True):
                wins = {t1_name: 0, t2_name: 0}
                results = {t1_name: [], t2_name: []}
                
                progress_bar = st.progress(0, text="Simulating...")
                for i in range(num_sims):
                    res1 = simulate_fast_innings(t1_stats, team_phase_stats[t1_name])
                    res2 = simulate_fast_innings(t2_stats, team_phase_stats[t2_name])
                    wins[t1_name if res1['score'] > res2['score'] else t2_name] += 1
                    results[t1_name].append(res1)
                    results[t2_name].append(res2)
                    progress_bar.progress((i + 1) / num_sims, text=f"Simulating... {i+1}/{num_sims}")
                
                st.subheader("Match Win Probabilities")
                win_p1 = (wins[t1_name] / num_sims) * 100
                st.metric(f"{t1_name} Win Probability", f"{win_p1:.1f}%")
                st.metric(f"{t2_name} Win Probability", f"{100-win_p1:.1f}%")

                st.subheader("Expected Averages & Betting Markets")
                for team in [t1_name, t2_name]:
                    with st.expander(f"View detailed stats for {team}"):
                        df = pd.DataFrame(results[team])
                        st.metric("Expected Runs", f"{df['score'].mean():.0f}")
                        st.metric("Expected Wickets", f"{df['wickets'].mean():.1f}")
                        st.metric("Avg. Runs at 1st Wicket Fall", f"{df['runs_at_1st_wicket'].mean():.0f}")
                        st.metric("Avg. Fours", f"{df['fours'].mean():.1f}")
                        st.metric("Avg. Sixes", f"{df['sixes'].mean():.1f}")
                        st.metric("Avg. Boundaries", f"{df['boundaries'].mean():.1f}")

        with tabs[4]: # Advanced Simulator
            st.header("🆚 Advanced Simulator (Phase-Based)")
            adv_cols = st.columns(2)
            adv_t1 = adv_cols[0].selectbox("Select Team 1", teams_players.keys(), key="adv_t1")
            adv_t2 = adv_cols[1].selectbox("Select Team 2", teams_players.keys(), index=1 if len(team_list) > 1 else 0, key="adv_t2")

            if st.button("Run Advanced Simulation", use_container_width=True):
                st.subheader(f"Simulating: {adv_t1} vs {adv_t2}")
                score1, wickets1 = 0, 0
                for ball in range(120):
                    if wickets1 >= 10: break
                    outcome = simulate_ball_by_phase(get_match_phase(ball//6), team_phase_stats[adv_t1])
                    if isinstance(outcome, int): score1 += outcome
                    else: wickets1 += 1
                
                score2, wickets2 = 0, 0
                for ball in range(120):
                    if wickets2 >= 10 or score2 > score1: break
                    outcome = simulate_ball_by_phase(get_match_phase(ball//6), team_phase_stats[adv_t2])
                    if isinstance(outcome, int): score2 += outcome
                    else: wickets2 += 1
                
                st.header("Match Result")
                res_cols = st.columns(2)
                res_cols[0].metric(f"{adv_t1} Score", f"{score1}/{wickets1}")
                res_cols[1].metric(f"{adv_t2} Score", f"{score2}/{wickets2}")
                st.subheader(f"🎉 {adv_t2 if score2 > score1 else adv_t1} wins!")

    else:
        st.warning("Could not process uploaded files. Please check file format and content.")
else:
    st.info("👋 Welcome! Upload JSON match logs to begin.")
