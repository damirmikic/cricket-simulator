import streamlit as st
import pandas as pd
import json
import os
import random
from collections import defaultdict
from io import BytesIO

# --- Core Data Processing Functions ---

def process_all_matches_and_players(uploaded_files):
    all_innings_data, player_batting_stats, player_bowling_stats = [], defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    teams_players, team_runs_per_ball = defaultdict(set), defaultdict(list)

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
                    
                    team_runs_per_ball[team].append(runs['batter'] if 'wickets' not in delivery else 'W')
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
            all_innings_data.append({'team': team, 'total_runs': innings_runs, 'run_rate': round((innings_runs * 6 / legal_balls) if legal_balls > 0 else 0, 2), 'wicket_rate': round((wickets_lost * 6 / legal_balls) if legal_balls > 0 else 0, 2)})

    innings_df = pd.DataFrame(all_innings_data) if all_innings_data else pd.DataFrame()
    batting_df = pd.DataFrame.from_dict(player_batting_stats, orient='index'); bowling_df = pd.DataFrame.from_dict(player_bowling_stats, orient='index')
    if not batting_df.empty: batting_df['strike_rate'] = round(batting_df['runs'] * 100 / batting_df['balls_faced'], 2).fillna(0)
    if not bowling_df.empty: bowling_df['economy'] = round(bowling_df['runs_conceded'] * 6 / bowling_df['balls_bowled'], 2).fillna(0); bowling_df['bowling_sr'] = round(bowling_df['balls_bowled'] / bowling_df['wickets'], 2).fillna(0)

    return innings_df, batting_df, bowling_df, {t: sorted(list(p)) for t, p in teams_players.items()}, team_runs_per_ball

def get_last_match_lineup(team, files):
    last_date, lineup = None, []
    for f in files:
        try:
            f.seek(0); data = json.load(f); info = data.get('info', {}); date = info.get('dates', [None])[0]
            if team in info.get('players', {}) and (not last_date or (date and date > last_date)):
                last_date, lineup = date, info['players'][team]
        except: continue
    return lineup

def summarize_match(file):
    try:
        file.seek(0); data = json.load(file); info = data.get('info', {})
        summary = { "Match": f"{info.get('teams', [])[0]} vs {info.get('teams', [])[1]}", "Venue": info.get('venue'), "Date": info.get('dates', ['N/A'])[0],
                    "Toss Winner": info.get('toss', {}).get('winner'), "Toss Decision": info.get('toss', {}).get('decision'),
                    "Winner": info.get('outcome', {}).get('winner'), "Player of the Match": info.get('player_of_match', ['N/A'])[0], "Innings": [] }
        for i, inning in enumerate(data.get('innings', [])):
            team = inning.get('team'); score, wickets = 0, 0; ball_by_ball = []
            batting_card = defaultdict(lambda: {'runs': 0, 'balls': 0})
            for over in inning.get('overs', []):
                for d in over.get('deliveries', []):
                    batter = d['batter']; runs = d['runs']['batter']
                    batting_card[batter]['runs'] += runs
                    if 'wides' not in d.get('extras', {}) and 'noballs' not in d.get('extras', {}):
                        batting_card[batter]['balls'] += 1
                    outcome = runs; event = f"{over['over']}.{len(ball_by_ball)%6+1}: {d['batter']} - {outcome} run(s)"
                    if 'wickets' in d: wickets+=1; outcome='W'; event += f" WICKET ({d['wickets'][0]['kind']})"
                    else: score += d['runs']['total']
                    ball_by_ball.append(event)
            summary["Innings"].append({"Team": team, "Score": score, "Wickets": wickets, "Scorecard": dict(batting_card), "Ball_by_Ball": ball_by_ball})
        return summary
    except Exception as e: return {"Error": str(e)}

# --- Simulation Functions ---
def simulate_fast_innings(team_stats):
    score, wickets, fours, sixes, runs_per_ball = 0, 0, 0, 0, []
    rr, wr = team_stats['run_rate'], team_stats['wicket_rate']

    for i in range(120):
        if wickets >= 10: break
        prob_wicket = wr / 6.0; prob_dot = max(0.1, 0.5 - (rr/20.0)); prob_four = max(0.05, (rr/60.0)); prob_six = max(0.02, (rr/90.0))
        outcomes, weights = ['WICKET', 0, 1, 2, 4, 6], [prob_wicket, prob_dot, 0.35, 0.05, prob_four, prob_six]
        outcome = random.choices(outcomes, [w / sum(weights) for w in weights])[0]

        runs_per_ball.append('W' if outcome == 'WICKET' else outcome)
        if outcome == 'WICKET': wickets += 1
        else:
            score += outcome;
            if outcome == 4: fours += 1
            if outcome == 6: sixes += 1

    return {"score": score, "wickets": wickets, "fours": fours, "sixes": sixes, "runs_per_ball": runs_per_ball}

def simulate_player_based_innings(batting_lineup, bowling_lineup, batting_df, bowling_df):
    score, wickets = 0, 0; on_strike, off_strike = batting_lineup[0], batting_lineup[1]; bowler_idx = 0
    bat_stats = {p: {'runs': 0, 'balls': 0} for p in batting_lineup}
    bowl_stats = {p: {'runs': 0, 'balls': 0, 'wickets': 0} for p in bowling_lineup}
    
    for i in range(120):
        if wickets >= 10: break
        bowler = bowling_lineup[bowler_idx % len(bowling_lineup)]
        batter_sr = batting_df.loc[on_strike]['strike_rate'] if on_strike in batting_df.index else 90
        bowler_econ = bowling_df.loc[bowler]['economy'] if bowler in bowling_df.index else 8.0
        bowler_sr = bowling_df.loc[bowler]['bowling_sr'] if bowler in bowling_df.index else 30

        prob_wicket = 1 / bowler_sr if bowler_sr > 0 else 0.03
        prob_four = (batter_sr/100)*(bowler_econ/8.0)/6.0; prob_six = prob_four/2
        outcomes, weights = ['WICKET',0,1,2,4,6], [prob_wicket,0.4,0.35,0.05,prob_four,prob_six]
        outcome = random.choices(outcomes, [w/sum(weights) for w in weights])[0]

        bat_stats[on_strike]['balls'] += 1; bowl_stats[bowler]['balls'] += 1
        if outcome == 'WICKET':
            wickets += 1; bowl_stats[bowler]['wickets'] += 1
            if wickets < 10: on_strike = batting_lineup[wickets + 1]
        else:
            score += outcome; bat_stats[on_strike]['runs'] += outcome; bowl_stats[bowler]['runs'] += outcome
            if outcome in [1, 3]: on_strike, off_strike = off_strike, on_strike
        
        if (i + 1) % 6 == 0 and wickets < 10: on_strike, off_strike = off_strike, on_strike; bowler_idx += 1
            
    return {"score": score, "wickets": wickets, "batting_card": bat_stats, "bowling_card": bowl_stats}

# --- Streamlit UI ---
st.set_page_config(page_title="Cricket Predictive Model", layout="wide")
st.title("🏏 Cricket Predictive Model")

if 'files' not in st.session_state: st.session_state.files = None
with st.sidebar:
    st.header("Upload Match Data"); uploaded_files = st.file_uploader("Upload JSON match logs", type=['json'], accept_multiple_files=True)
    if uploaded_files: st.session_state.files = uploaded_files
    if st.button("Clear Uploaded Files"): st.session_state.files = None; st.rerun()

if st.session_state.files:
    innings_df, batting_df, bowling_df, teams_players, team_runs_per_ball = process_all_matches_and_players(st.session_state.files)
    
    if not innings_df.empty:
        team_stats = innings_df.groupby('team').agg(run_rate=('run_rate', 'mean'), wicket_rate=('wicket_rate', 'mean')).round(2).reset_index()

        tab1, tab2, tab3, tab4 = st.tabs(["📝 Match Summary", "📊 Team & Player Stats", "⚡ Fast Simulator", "🏏 Advanced Simulator"])
        
        with tab1:
            st.header("📝 Match Summary")
            file_map = {f.name: f for f in st.session_state.files}
            selected_file = st.selectbox("Select a match to summarize", file_map.keys())
            if selected_file:
                summary = summarize_match(file_map[selected_file])
                if "Error" in summary: st.error(f"Failed to process file: {summary['Error']}")
                else: st.json(summary, expanded=False)

        with tab2:
            st.header("📊 Team & Player Stats"); st.dataframe(team_stats)
            with st.expander("View Team Runs Per Ball"): st.json({k: ' '.join(map(str, v)) for k, v in team_runs_per_ball.items()})
            st.header("Player Performance"); st.subheader("Batting Stats"); st.dataframe(batting_df.sort_values('runs', ascending=False))
            st.subheader("Bowling Stats"); st.dataframe(bowling_df.sort_values('wickets', ascending=False))

        with tab3:
            st.header("⚡ Fast Simulator"); team_list = list(teams_players.keys())
            c1,c2 = st.columns(2); t1_name = c1.selectbox("Team 1", team_list, key="sim_t1"); t2_name = c2.selectbox("Team 2", team_list, index=1 if len(team_list)>1 else 0, key="sim_t2")
            
            num_sims = st.number_input("Simulations?", 1, 10000, 100)
            if st.button("▶️ Run Fast Simulation", use_container_width=True):
                t1_stats=team_stats[team_stats['team']==t1_name].iloc[0]; t2_stats=team_stats[team_stats['team']==t2_name].iloc[0]
                wins={t1_name:0,t2_name:0}; results={t1_name:[],t2_name:[]}
                
                for i in range(num_sims):
                    res1 = simulate_fast_innings(t1_stats); res2 = simulate_fast_innings(t2_stats)
                    wins[t1_name if res1['score'] > res2['score'] else t2_name] += 1
                    results[t1_name].append(res1); results[t2_name].append(res2)
                
                st.subheader("Win Probabilities"); win_p1=(wins[t1_name]/num_sims)*100
                st.metric(f"{t1_name} Win Prob.", f"{win_p1:.1f}%"); st.metric(f"{t2_name} Win Prob.", f"{100-win_p1:.1f}%")

                df1=pd.DataFrame(results[t1_name]); df2=pd.DataFrame(results[t2_name])
                with st.expander("View Betting Markets & Runs Per Ball"):
                    st.metric("Expected Runs", f"{df1['score'].mean():.0f} - {df2['score'].mean():.0f}")
                    st.text(f"{t1_name} (Sample RPB): {' '.join(map(str, df1.iloc[0]['runs_per_ball']))}")
                    st.text(f"{t2_name} (Sample RPB): {' '.join(map(str, df2.iloc[0]['runs_per_ball']))}")

        with tab4:
            st.header("🏏 Advanced Simulator"); c1,c2=st.columns(2)
            team1_name = c1.selectbox("Batting Team", teams_players.keys(), key="adv_t1"); team2_name = c2.selectbox("Bowling Team", teams_players.keys(), index=1 if len(teams_players.keys())>1 else 0, key="adv_t2")
            
            l1,l2=st.columns(2)
            lineup1=l1.multiselect(f"{team1_name} Batting", options=teams_players[team1_name], default=get_last_match_lineup(team1_name, st.session_state.files)[:11], max_selections=11)
            lineup2=l2.multiselect(f"{team2_name} Bowling", options=teams_players[team2_name], default=get_last_match_lineup(team2_name, st.session_state.files)[:11], max_selections=11)
            
            if st.button("▶️ Run Player-Based Simulation", use_container_width=True):
                if len(lineup1)!=11 or len(lineup2)!=11: st.error("Please select 11 players for each team.")
                else:
                    result=simulate_player_based_innings(lineup1, lineup2, batting_df, bowling_df)
                    st.subheader(f"Innings Result: {result['score']}/{result['wickets']}")
                    c1,c2=st.columns(2)
                    c1.dataframe(pd.DataFrame.from_dict(result['batting_card'], orient='index').reset_index().rename(columns={'index':'Batter'}))
                    c2.dataframe(pd.DataFrame.from_dict(result['bowling_card'], orient='index').reset_index().rename(columns={'index':'Bowler'}))

    else:
        st.warning("Could not process uploaded files.")
else:
    st.info("👋 Welcome! Upload JSON match logs to begin.")
