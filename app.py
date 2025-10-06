import streamlit as st
import pandas as pd
import json
import os
import random
from collections import defaultdict
from io import BytesIO

# --- Core Data Processing Functions ---

def get_match_phase(over_num):
    if over_num <= 5: return 'Powerplay'
    if over_num <= 14: return 'Middle'
    return 'Death'

def process_all_matches_and_players(uploaded_files):
    all_innings_data, player_batting_stats, player_bowling_stats = [], defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    teams_players, team_phase_stats = defaultdict(set), defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for f in uploaded_files:
        try:
            f.seek(0); data = json.load(f)
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError) as e:
            st.warning(f"Could not process file {getattr(f, 'name', 'N/A')}: {e}")
            continue

        info = data.get('info', {})
        for team, players in info.get('players', {}).items(): teams_players[team].update(players)
        
        for inning in data.get('innings', []):
            team = inning.get('team')
            for over in inning.get('overs', []):
                for delivery in over.get('deliveries', []):
                    batter, bowler, runs = delivery['batter'], delivery['bowler'], delivery['runs']
                    is_legal = 'wides' not in delivery.get('extras', {}) and 'noballs' not in delivery.get('extras', {})
                    
                    if is_legal:
                        player_batting_stats[batter]['balls_faced'] += 1
                        player_bowling_stats[bowler]['balls_bowled'] += 1
                    player_batting_stats[batter]['runs'] += runs['batter']
                    player_bowling_stats[bowler]['runs_conceded'] += runs['total']
                    if 'wickets' in delivery and delivery['wickets'][0]['kind'] not in ['run out', 'retired hurt']:
                        player_bowling_stats[bowler]['wickets'] += 1

            innings_runs = sum(p['runs']['total'] for o in inning.get('overs', []) for p in o.get('deliveries', []))
            legal_balls = sum(1 for o in inning.get('overs', []) for d in o.get('deliveries', []) if 'wides' not in d.get('extras', {}) and 'noballs' not in d.get('extras', {}))
            wickets_lost = sum(1 for o in inning.get('overs', []) for d in o.get('deliveries', []) if 'wickets' in d)
            all_innings_data.append({
                'team': team, 'total_runs': innings_runs,
                'run_rate': round((innings_runs * 6 / legal_balls) if legal_balls > 0 else 0, 2),
                'wicket_rate': round((wickets_lost * 6 / legal_balls) if legal_balls > 0 else 0, 2)
            })

    innings_df = pd.DataFrame(all_innings_data) if all_innings_data else pd.DataFrame()
    batting_df = pd.DataFrame.from_dict(player_batting_stats, orient='index')
    bowling_df = pd.DataFrame.from_dict(player_bowling_stats, orient='index')
    
    if not batting_df.empty:
        batting_df['strike_rate'] = round(batting_df['runs'] * 100 / batting_df['balls_faced'], 2).fillna(0)
    if not bowling_df.empty:
        bowling_df['economy'] = round(bowling_df['runs_conceded'] * 6 / bowling_df['balls_bowled'], 2).fillna(0)
        bowling_df['bowling_sr'] = round(bowling_df['balls_bowled'] / bowling_df['wickets'], 2).fillna(0)

    return innings_df, batting_df, bowling_df, {t: sorted(list(p)) for t, p in teams_players.items()}

def get_last_match_lineup(team, files):
    last_date, lineup = None, []
    for f in files:
        try:
            f.seek(0); data = json.load(f); info = data.get('info', {}); date = info.get('dates', [None])[0]
            if team in info.get('players', {}) and (not last_date or (date and date > last_date)):
                last_date, lineup = date, info['players'][team]
        except: continue
    return lineup

# --- Simulation Functions ---
def simulate_fast_innings(team_stats):
    score, wickets, fours, sixes = 0, 0, 0, 0
    runs_at_1st_wicket, runs_first_over, runs_powerplay = -1, 0, 0
    rr, wr = team_stats['run_rate'], team_stats['wicket_rate']

    for i in range(120):
        if wickets >= 10: break
        prob_wicket = wr / 6.0; prob_dot = max(0.1, 0.5 - (rr/20.0)); prob_four = max(0.05, (rr/60.0)); prob_six = max(0.02, (rr/90.0))
        outcomes, weights = ['WICKET', 0, 1, 2, 4, 6], [prob_wicket, prob_dot, 0.35, 0.05, prob_four, prob_six]
        outcome = random.choices(outcomes, [w / sum(weights) for w in weights])[0]

        if outcome == 'WICKET':
            if runs_at_1st_wicket == -1: runs_at_1st_wicket = score
            wickets += 1
        else:
            score += outcome
            if outcome == 4: fours += 1
            if outcome == 6: sixes += 1
        if i < 6: runs_first_over = score
        if i < 36: runs_powerplay = score

    return {"score": score, "wickets": wickets, "fours": fours, "sixes": sixes, "boundaries": fours + sixes,
            "runs_at_1st_wicket": runs_at_1st_wicket if runs_at_1st_wicket != -1 else score, 
            "runs_first_over": runs_first_over, "runs_powerplay": runs_powerplay}

def simulate_player_based_innings(batting_lineup, bowling_lineup, batting_df, bowling_df):
    score, wickets, fours, sixes = 0, 0, 0, 0
    runs_at_1st_wicket, runs_first_over, runs_powerplay = -1, 0, 0
    
    bat_stats = {p: {'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0} for p in batting_lineup}
    bowl_stats = {p: {'runs': 0, 'balls': 0, 'wickets': 0} for p in bowling_lineup}
    
    on_strike, off_strike = batting_lineup[0], batting_lineup[1]
    bowler_idx = 0

    for i in range(120):
        if wickets >= 10: break
        bowler = bowling_lineup[bowler_idx % len(bowling_lineup)]
        
        batter_sr = batting_df.loc[on_strike]['strike_rate'] if on_strike in batting_df.index else 90
        bowler_econ = bowling_df.loc[bowler]['economy'] if bowler in bowling_df.index else 8.0
        bowler_sr = bowling_df.loc[bowler]['bowling_sr'] if bowler in bowling_df.index else 30

        prob_wicket = 1 / bowler_sr if bowler_sr > 0 else 0.03
        prob_four = (batter_sr / 100) * (bowler_econ / 8.0) / 6.0; prob_six = prob_four / 2
        outcomes, weights = ['WICKET', 0, 1, 2, 4, 6], [prob_wicket, 0.4, 0.35, 0.05, prob_four, prob_six]
        outcome = random.choices(outcomes, [w / sum(weights) for w in weights])[0]
        
        bat_stats[on_strike]['balls'] += 1
        bowl_stats[bowler]['balls'] += 1

        if outcome == 'WICKET':
            if runs_at_1st_wicket == -1: runs_at_1st_wicket = score
            wickets += 1; bowl_stats[bowler]['wickets'] += 1
            if wickets < 10: on_strike = batting_lineup[wickets + 1]
        else:
            score += outcome; bat_stats[on_strike]['runs'] += outcome; bowl_stats[bowler]['runs'] += outcome
            if outcome == 4: fours += 1; bat_stats[on_strike]['fours'] += 1
            if outcome == 6: sixes += 1; bat_stats[on_strike]['sixes'] += 1
            if outcome in [1, 3]: on_strike, off_strike = off_strike, on_strike
        
        if (i + 1) % 6 == 0 and wickets < 10:
            on_strike, off_strike = off_strike, on_strike; bowler_idx += 1
        if i < 6: runs_first_over = score
        if i < 36: runs_powerplay = score

    return {"team_stats": {"score": score, "wickets": wickets, "fours": fours, "sixes": sixes, "boundaries": fours + sixes,
            "runs_at_1st_wicket": runs_at_1st_wicket if runs_at_1st_wicket != -1 else score,
            "runs_first_over": runs_first_over, "runs_powerplay": runs_powerplay},
            "batting_card": bat_stats, "bowling_card": bowl_stats}

# --- Streamlit UI ---
st.set_page_config(page_title="Cricket Predictive Model", layout="wide")
st.title("🏏 Cricket Predictive Model")

# Initialize session state for file persistence
if 'files' not in st.session_state:
    st.session_state.files = None

# Sidebar for file uploading and clearing
with st.sidebar:
    st.header("Upload Match Data")
    uploaded_files = st.file_uploader("Upload JSON match logs", type=['json'], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.files = uploaded_files
    
    if st.button("Clear Uploaded Files"):
        st.session_state.files = None
        st.rerun() # Rerun the app to reflect the cleared state

# Main app logic
if st.session_state.files:
    innings_df, batting_df, bowling_df, teams_players = process_all_matches_and_players(st.session_state.files)
    
    if not innings_df.empty:
        team_stats = innings_df.groupby('team').agg(run_rate=('run_rate', 'mean'), wicket_rate=('wicket_rate', 'mean')).round(2).reset_index()

        tab1, tab2, tab3 = st.tabs(["📊 Team & Player Stats", "⚡ Fast Simulator (Team-Based)", "🏏 Advanced Simulator (Player-Based)"])

        with tab2:
            st.header("⚡ Fast Match Simulator"); team_list = list(teams_players.keys())
            c1,c2 = st.columns(2); t1_name = c1.selectbox("Select Team 1", team_list, key="sim_t1"); t2_name = c2.selectbox("Select Team 2", team_list, index=1 if len(team_list)>1 else 0, key="sim_t2")

            st.subheader("Team Parameters")
            p1,p2=st.columns(2); t1_stats=team_stats[team_stats['team']==t1_name].iloc[0]; t2_stats=team_stats[team_stats['team']==t2_name].iloc[0]
            p1.metric(f"{t1_name} Avg Run Rate", f"{t1_stats['run_rate']:.2f}"); p1.metric(f"{t1_name} Avg Wicket Rate", f"{t1_stats['wicket_rate']:.2f}")
            p2.metric(f"{t2_name} Avg Run Rate", f"{t2_stats['run_rate']:.2f}"); p2.metric(f"{t2_name} Avg Wicket Rate", f"{t2_stats['wicket_rate']:.2f}")

            num_sims = st.number_input("How many simulations?", 1, 10000, 100)
            if st.button("▶️ Run Fast Simulation", use_container_width=True):
                wins={t1_name:0,t2_name:0}; results={t1_name:[],t2_name:[]}
                progress_bar = st.progress(0, text="Simulating...")
                for i in range(num_sims):
                    res1 = simulate_fast_innings(t1_stats); res2 = simulate_fast_innings(t2_stats)
                    wins[t1_name if res1['score'] > res2['score'] else t2_name] += 1
                    results[t1_name].append(res1); results[t2_name].append(res2)
                    progress_bar.progress((i+1)/num_sims, text=f"Simulating... {i+1}/{num_sims}")
                
                st.subheader("Match Win Probabilities")
                win_p1=(wins[t1_name]/num_sims)*100; st.metric(f"{t1_name} Win Probability", f"{win_p1:.1f}%"); st.metric(f"{t2_name} Win Probability", f"{100-win_p1:.1f}%")

                df1=pd.DataFrame(results[t1_name]); df2=pd.DataFrame(results[t2_name])
                with st.expander("View Detailed Betting Markets"):
                    for team, df in [(t1_name, df1), (t2_name, df2)]:
                        st.subheader(f"'{team}' Market Predictions")
                        c1,c2,c3=st.columns(3)
                        c1.metric("Expected Runs", f"{df['score'].mean():.0f}"); c2.metric("Expected Wickets", f"{df['wickets'].mean():.1f}"); c3.metric("Avg Runs at 1st Wicket", f"{df['runs_at_1st_wicket'].mean():.0f}")
                        c1.metric("Avg Fours", f"{df['fours'].mean():.1f}"); c2.metric("Avg Sixes", f"{df['sixes'].mean():.1f}"); c3.metric("Avg Boundaries", f"{df['boundaries'].mean():.1f}")
                        c1.metric("Avg Runs in First Over", f"{df['runs_first_over'].mean():.1f}"); c2.metric("Avg Runs in Powerplay", f"{df['runs_powerplay'].mean():.1f}")

                    st.subheader("Team Comparison Markets")
                    st.metric("Prob. of Team 1 having more Fours", f"{(df1['fours'] > df2['fours']).mean() * 100:.1f}%")
                    st.metric("Prob. of Team 1 having more Sixes", f"{(df1['sixes'] > df2['sixes']).mean() * 100:.1f}%")
                    st.metric("Prob. of Team 1 having more Boundaries", f"{(df1['boundaries'] > df2['boundaries']).mean() * 100:.1f}%")

        with tab3:
            st.header("🏏 Advanced Simulator & Lineup Creator")
            c1,c2=st.columns(2); team1_name=c1.selectbox("Select Batting Team", teams_players.keys(),key="adv_t1"); team2_name=c2.selectbox("Select Bowling Team", teams_players.keys(),index=1 if len(teams_players.keys())>1 else 0, key="adv_t2")
            
            l1, l2 = st.columns(2)
            lineup1=l1.multiselect(f"{team1_name} (Batting)", options=teams_players[team1_name], default=get_last_match_lineup(team1_name, st.session_state.files)[:11], max_selections=11)
            lineup2=l2.multiselect(f"{team2_name} (Bowling)", options=teams_players[team2_name], default=get_last_match_lineup(team2_name, st.session_state.files)[:11], max_selections=11)

            if st.button("▶️ Run Player-Based Simulation", use_container_width=True):
                if len(lineup1)!=11 or len(lineup2)!=11: st.error("Please select exactly 11 players for each team.")
                else:
                    result=simulate_player_based_innings(lineup1, lineup2, batting_df, bowling_df)
                    st.subheader(f"Innings Result: {result['team_stats']['score']}/{result['team_stats']['wickets']}")
                    
                    with st.expander("View Innings Markets"):
                        stats = result['team_stats']
                        c1,c2,c3=st.columns(3)
                        c1.metric("Total Fours", stats['fours']); c2.metric("Total Sixes", stats['sixes']); c3.metric("Total Boundaries", stats['boundaries'])
                        c1.metric("Runs at 1st Wicket", stats['runs_at_1st_wicket']); c2.metric("Runs in First Over", stats['runs_first_over']); c3.metric("Runs in Powerplay", stats['runs_powerplay'])
                    
                    st.subheader("Player Performance Scorecards")
                    bat_card = pd.DataFrame.from_dict(result['batting_card'], orient='index').reset_index().rename(columns={'index':'Batter'})
                    bowl_card = pd.DataFrame.from_dict(result['bowling_card'], orient='index').reset_index().rename(columns={'index':'Bowler'})
                    c1,c2=st.columns(2); c1.dataframe(bat_card); c2.dataframe(bowl_card)
    else:
        st.warning("Could not process uploaded files.")

else:
    st.info("👋 Welcome! Upload JSON match logs to begin.")
