import streamlit as st
import pandas as pd
import json
import os
import random
from collections import defaultdict
from io import BytesIO

# --- Core Data Processing Functions ---

def get_match_phase(over_num):
    """Determines the phase of the match based on the over number."""
    if over_num <= 5: return 'Powerplay'
    if over_num <= 14: return 'Middle'
    return 'Death'

def process_all_matches_and_players(uploaded_files):
    all_innings_data = []
    player_batting_stats = defaultdict(lambda: defaultdict(int))
    player_bowling_stats = defaultdict(lambda: defaultdict(int))
    teams_players = defaultdict(set)
    team_phase_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

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
                    if is_legal: team_phase_stats[team][phase]['balls'] += 1
                    if 'wickets' in delivery: team_phase_stats[team][phase]['wickets'] += 1
                    
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
                'run_rate': round((innings_runs * 6 / innings_balls) if innings_balls > 0 else 0, 2),
                'wicket_rate': round((innings_wickets * 6 / innings_balls) if innings_balls > 0 else 0, 2),
                'is_win': is_win
            })

    innings_df = pd.DataFrame(all_innings_data) if all_innings_data else pd.DataFrame()
    batting_df = pd.DataFrame.from_dict(player_batting_stats, orient='index')
    bowling_df = pd.DataFrame.from_dict(player_bowling_stats, orient='index')
    
    if not batting_df.empty:
        batting_df['strike_rate'] = round(batting_df['runs'] * 100 / batting_df['balls_faced'], 2).fillna(0)
        batting_df['avg'] = round(batting_df['runs'] / (batting_df.index.map(bowling_df['wickets']).fillna(0)), 2).fillna(0) # A bit simplistic
    if not bowling_df.empty:
        bowling_df['economy'] = round(bowling_df['runs_conceded'] * 6 / bowling_df['balls_bowled'], 2).fillna(0)
        bowling_df['bowling_sr'] = round(bowling_df['balls_bowled'] / bowling_df['wickets'], 2).fillna(0)

    teams_players = {team: sorted(list(players)) for team, players in teams_players.items()}
    return innings_df, batting_df, bowling_df, teams_players, team_phase_stats

def get_last_match_lineup(team, uploaded_files):
    last_date, lineup = None, []
    for f in uploaded_files:
        try:
            f.seek(0); data = json.load(f); info = data.get('info', {})
            match_date = info.get('dates', [None])[0]
            if team in info.get('players', {}):
                if not last_date or (match_date and match_date > last_date):
                    last_date, lineup = match_date, info['players'][team]
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError): continue
    return lineup

# --- Simulation Functions ---
def simulate_fast_innings(team_stats):
    score, wickets, runs_per_ball, match_flow = 0, 0, [], []
    rr, wr = team_stats['run_rate'], team_stats['wicket_rate']
    
    for i in range(120):
        if wickets >= 10: break
        prob_wicket = wr / 6.0; prob_dot = max(0.1, 0.5 - (rr / 20.0)); prob_four = max(0.05, (rr / 60.0)); prob_six = max(0.02, (rr / 90.0))
        outcomes = ['WICKET', 0, 1, 2, 4, 6]; weights = [prob_wicket, prob_dot, 0.35, 0.05, prob_four, prob_six]
        outcome = random.choices(outcomes, [w / sum(weights) for w in weights])[0]
        
        over_str = f"{i//6}.{i%6+1}"
        if outcome == 'WICKET':
            wickets += 1; runs_per_ball.append('W'); match_flow.append(f"{over_str}: WICKET! Score: {score}/{wickets}")
        else:
            score += outcome; runs_per_ball.append(outcome); match_flow.append(f"{over_str}: {outcome} run(s). Score: {score}/{wickets}")
            
    return {"score": score, "wickets": wickets, "runs_per_ball": runs_per_ball, "match_flow": match_flow}

def simulate_player_based_innings(batting_lineup, bowling_lineup, batting_df, bowling_df):
    score, wickets = 0, 0
    match_flow, runs_per_ball = [], []
    batter_idx, bowler_idx = 0, 0
    on_strike, off_strike = batting_lineup[0], batting_lineup[1]
    
    for i in range(120):
        if wickets >= 10: break
        bowler = bowling_lineup[bowler_idx % len(bowling_lineup)]
        
        batter_sr = batting_df.loc[on_strike]['strike_rate'] if on_strike in batting_df.index else 90
        bowler_econ = bowling_df.loc[bowler]['economy'] if bowler in bowling_df.index else 8.0
        bowler_sr = bowling_df.loc[bowler]['bowling_sr'] if bowler in bowling_df.index else 30

        prob_wicket = 1 / bowler_sr if bowler_sr > 0 else 0.03
        prob_four = (batter_sr / 100) * (bowler_econ / 8.0) / 6.0
        prob_six = (batter_sr / 100) * (bowler_econ / 8.0) / 12.0
        
        outcomes = ['WICKET', 0, 1, 2, 4, 6]; weights = [prob_wicket, 0.4, 0.35, 0.05, prob_four, prob_six]
        outcome = random.choices(outcomes, [w / sum(weights) for w in weights])[0]
        
        over_str = f"{i//6}.{i%6+1}"
        match_flow.append(f"{over_str}: {bowler} to {on_strike}...")
        
        if outcome == 'WICKET':
            wickets += 1; runs_per_ball.append('W');
            match_flow.append(f"OUT! Score: {score}/{wickets}")
            if wickets < 10: on_strike = batting_lineup[wickets + 1]
        else:
            score += outcome; runs_per_ball.append(outcome)
            match_flow.append(f"{outcome} run(s). Score: {score}/{wickets}")
            if outcome in [1, 3]: on_strike, off_strike = off_strike, on_strike
        
        if (i + 1) % 6 == 0 and wickets < 10:
            on_strike, off_strike = off_strike, on_strike
            bowler_idx += 1
            match_flow.append("--- End of Over ---")

    return {"score": score, "wickets": wickets, "runs_per_ball": runs_per_ball, "match_flow": match_flow}


# --- Streamlit UI ---
st.set_page_config(page_title="Cricket Predictive Model", layout="wide")
st.title("🏏 Cricket Predictive Model")

if 'uploaded_files' not in st.session_state: st.session_state.uploaded_files = None
uploaded_files = st.sidebar.file_uploader("Upload JSON match logs", type=['json'], accept_multiple_files=True)
if uploaded_files: st.session_state.uploaded_files = uploaded_files

if st.session_state.uploaded_files:
    data_load_state = st.text('Loading and processing data...')
    innings_df, batting_df, bowling_df, teams_players, team_phase_stats = process_all_matches_and_players(st.session_state.uploaded_files)
    data_load_state.text('Processing complete!')
    
    if not innings_df.empty:
        team_stats = innings_df.groupby('team').agg(run_rate=('run_rate', 'mean'), wicket_rate=('wicket_rate', 'mean')).round(2).reset_index()

        tab1, tab2, tab3 = st.tabs(["📊 Team & Player Stats", "⚡ Fast Simulator (Team-Based)", "🏏 Advanced Simulator (Player-Based)"])

        with tab1:
            st.header("Team Performance Summary"); st.dataframe(team_stats)
            st.header("Player Performance"); st.subheader("Batting Stats"); st.dataframe(batting_df.sort_values('runs', ascending=False))
            st.subheader("Bowling Stats"); st.dataframe(bowling_df.sort_values('wickets', ascending=False))

        with tab2:
            st.header("⚡ Fast Match Simulator (Based on Team Averages)")
            t1_name, t2_name = st.columns(2)
            team_list = list(teams_players.keys())
            t1 = t1_name.selectbox("Select Team 1", team_list, key="sim_t1")
            t2 = t2_name.selectbox("Select Team 2", team_list, index=1 if len(team_list) > 1 else 0, key="sim_t2")

            if st.button("▶️ Run Fast Simulation", use_container_width=True):
                t1_stats = team_stats[team_stats['team'] == t1].iloc[0]
                t2_stats = team_stats[team_stats['team'] == t2].iloc[0]
                
                res1 = simulate_fast_innings(t1_stats)
                res2 = simulate_fast_innings(t2_stats)
                
                st.subheader("Match Result")
                st.metric(f"{t1} Score", f"{res1['score']}/{res1['wickets']}")
                st.metric(f"{t2} Score", f"{res2['score']}/{res2['wickets']}")
                st.subheader(f"🎉 {t2 if res2['score'] > res1['score'] else t1} wins!")

                c1, c2 = st.columns(2)
                with c1.expander(f"View {t1} Innings Flow"): st.text_area("", "\n".join(res1['match_flow']), height=300)
                with c2.expander(f"View {t2} Innings Flow"): st.text_area("", "\n".join(res2['match_flow']), height=300)

        with tab3:
            st.header("🏏 Advanced Simulator (Based on Player Stats)")
            
            c1, c2 = st.columns(2)
            team1_name = c1.selectbox("Select Batting Team", teams_players.keys(), key="adv_t1")
            team2_name = c2.selectbox("Select Bowling Team", teams_players.keys(), index=1 if len(teams_players.keys()) > 1 else 0, key="adv_t2")

            st.subheader("Create Lineups")
            lineup1_col, lineup2_col = st.columns(2)
            
            with lineup1_col:
                last_match1 = get_last_match_lineup(team1_name, st.session_state.uploaded_files)[:11]
                lineup1 = st.multiselect(f"{team1_name} (Batting)", options=teams_players[team1_name], default=last_match1, max_selections=11)
            
            with lineup2_col:
                last_match2 = get_last_match_lineup(team2_name, st.session_state.uploaded_files)[:11]
                lineup2 = st.multiselect(f"{team2_name} (Bowling)", options=teams_players[team2_name], default=last_match2, max_selections=11)

            if st.button("▶️ Run Player-Based Simulation", use_container_width=True):
                if len(lineup1) != 11 or len(lineup2) != 11:
                    st.error("Please select exactly 11 players for each team.")
                else:
                    st.subheader(f"Simulating Innings: {team1_name}")
                    result = simulate_player_based_innings(lineup1, lineup2, batting_df, bowling_df)
                    
                    st.header("Innings Result")
                    st.metric(f"{team1_name} Score", f"{result['score']}/{result['wickets']}")
                    with st.expander("View Full Match Flow"):
                        st.text_area("", "\n".join(result['match_flow']), height=400)
    else:
        st.warning("Could not process uploaded files.")
else:
    st.info("👋 Welcome! Upload JSON match logs to begin.")
