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
    teams_players, team_runs_per_ball, league_data = defaultdict(set), defaultdict(list), []

    for f in uploaded_files:
        try:
            f.seek(0); data = json.load(f)
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError) as e:
            st.warning(f"Could not process file {getattr(f, 'name', 'N/A')}: {e}"); continue

        info = data.get('info', {})
        league = info.get('competition', 'Unknown League')
        season = info.get('season', 'Unknown Season')
        
        for team, players in info.get('players', {}).items(): teams_players[team].update(players)
        
        for inning in data.get('innings', []):
            team = inning.get('team')
            runs_inning, wickets_inning, fours, sixes = 0, 0, 0, 0
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
                    if runs['batter'] == 4: fours += 1
                    if runs['batter'] == 6: sixes += 1
                    if 'wickets' in delivery and delivery['wickets'][0]['kind'] not in ['run out', 'retired hurt']:
                        player_bowling_stats[bowler]['wickets'] += 1
                    runs_inning += runs['total']
                    if 'wickets' in delivery: wickets_inning += 1

            innings_runs = sum(p['runs']['total'] for o in inning.get('overs', []) for p in o.get('deliveries', []))
            legal_balls = sum(1 for o in inning.get('overs', []) for d in o.get('deliveries', []) if 'wides' not in d.get('extras', {}) and 'noballs' not in d.get('extras', {}))
            wickets_lost = sum(1 for o in inning.get('overs', []) for d in o.get('deliveries', []) if 'wickets' in d)
            all_innings_data.append({'team': team, 'total_runs': innings_runs, 'run_rate': round((innings_runs * 6 / legal_balls) if legal_balls > 0 else 0, 2), 'wicket_rate': round((wickets_lost * 6 / legal_balls) if legal_balls > 0 else 0, 2)})
            league_data.append({'league': league, 'season': season, 'team': team, 'runs': runs_inning, 'wickets': wickets_inning, 'fours': fours, 'sixes': sixes})

    innings_df = pd.DataFrame(all_innings_data) if all_innings_data else pd.DataFrame()
    batting_df = pd.DataFrame.from_dict(player_batting_stats, orient='index'); bowling_df = pd.DataFrame.from_dict(player_bowling_stats, orient='index')
    if not batting_df.empty: batting_df['strike_rate'] = round(batting_df['runs'] * 100 / batting_df['balls_faced'], 2).fillna(0)
    if not bowling_df.empty: bowling_df['economy'] = round(bowling_df['runs_conceded'] * 6 / bowling_df['balls_bowled'], 2).fillna(0); bowling_df['bowling_sr'] = round(bowling_df['balls_bowled'] / bowling_df['wickets'], 2).fillna(0)

    return innings_df, batting_df, bowling_df, {t: sorted(list(p)) for t, p in teams_players.items()}, team_runs_per_ball, pd.DataFrame(league_data)

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
        summary = { "info": { "Match": f"{info.get('teams', [])[0]} vs {info.get('teams', [])[1]}", "Venue": info.get('venue'), "Date": info.get('dates', ['N/A'])[0],
                              "Toss Winner": info.get('toss', {}).get('winner'), "Toss Decision": info.get('toss', {}).get('decision'),
                              "Winner": info.get('outcome', {}).get('winner'), "Player of the Match": info.get('player_of_match', ['N/A'])[0]}, "innings": [] }
        for i, inning_data in enumerate(data.get('innings', [])):
            team = inning_data.get('team'); score, wickets, fours, sixes = 0, 0, 0, 0
            runs_at_1st_wicket, runs_first_over, highest_partnership = -1, 0, 0
            phase_runs = {'Powerplay': 0, 'Middle': 0, 'Death': 0}
            batting_card = defaultdict(lambda: {'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0})
            
            for over in inning_data.get('overs', []):
                phase = 'Powerplay' if over['over'] < 6 else ('Middle' if over['over'] < 15 else 'Death')
                for d in over.get('deliveries', []):
                    batter, runs = d['batter'], d['runs']['batter']
                    batting_card[batter]['runs'] += runs
                    if 'wides' not in d.get('extras', {}) and 'noballs' not in d.get('extras', {}): batting_card[batter]['balls'] += 1
                    if runs == 4: batting_card[batter]['fours'] += 1; fours += 1
                    if runs == 6: batting_card[batter]['sixes'] += 1; sixes += 1
                    
                    score += d['runs']['total']; phase_runs[phase] += d['runs']['total']
                    if 'wickets' in d: 
                        if wickets == 0: runs_at_1st_wicket = score
                        wickets += 1
                    if over['over'] == 0: runs_first_over = score
            
            summary["innings"].append({"team": team, "score": score, "wickets": wickets, "fours": fours, "sixes": sixes, "boundaries": fours + sixes,
                                       "runs_at_1st_wicket": runs_at_1st_wicket if runs_at_1st_wicket != -1 else score, "runs_first_over": runs_first_over,
                                       "phase_runs": phase_runs, "scorecard": {k: dict(v) for k, v in batting_card.items()}})
        return summary
    except Exception as e: return {"error": str(e)}

# --- Streamlit UI ---
st.set_page_config(page_title="Cricket Predictive Model", layout="wide")
st.title("🏏 Cricket Predictive Model")

if 'files' not in st.session_state: st.session_state.files = None
with st.sidebar:
    st.header("Upload Match Data"); uploaded_files = st.file_uploader("Upload JSON match logs", type=['json'], accept_multiple_files=True)
    if uploaded_files: st.session_state.files = uploaded_files
    if st.button("Clear Uploaded Files"): st.session_state.files = None; st.rerun()

if st.session_state.files:
    innings_df, batting_df, bowling_df, teams_players, team_runs_per_ball, league_df = process_all_matches_and_players(st.session_state.files)
    
    if not innings_df.empty:
        team_stats = innings_df.groupby('team').agg(run_rate=('run_rate', 'mean'), wicket_rate=('wicket_rate', 'mean')).round(2).reset_index()

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Match Summary", "📊 Team & Player Stats", "🏆 League Analysis", "⚡ Fast Simulator", "🏏 Advanced Simulator"])
        
        with tab1:
            st.header("📝 Match Summary")
            file_map = {f.name: f for f in st.session_state.files}
            selected_file = st.selectbox("Select a match to summarize", file_map.keys())
            if selected_file:
                summary = summarize_match(file_map[selected_file])
                if "error" in summary: st.error(f"Failed to process file: {summary['error']}")
                else:
                    st.subheader(summary['info']['Match']); st.caption(f"{summary['info']['Venue']} | {summary['info']['Date']}")
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Winner", summary['info']['Winner']); c2.metric("Toss Winner", summary['info']['Toss Winner'])
                    c3.metric("Toss Decision", summary['info']['Toss Decision']); c4.metric("Player of the Match", summary['info']['Player of the Match'])
                    
                    for inning in summary['innings']:
                        with st.expander(f"**{inning['team']} Innings: {inning['score']}/{inning['wickets']}**", expanded=True):
                            sc1,sc2,sc3,sc4 = st.columns(4)
                            sc1.metric("Fours", inning['fours']); sc2.metric("Sixes", inning['sixes']); sc3.metric("Boundaries", inning['boundaries'])
                            sc4.metric("Runs at 1st Wicket", inning['runs_at_1st_wicket'])
                            
                            st.dataframe(pd.DataFrame.from_dict(inning['scorecard'], orient='index'))
                            st.write("**Runs by Phase**"); st.json(inning['phase_runs'])

        with tab2:
            st.header("📊 Aggregated Team & Player Stats"); st.dataframe(team_stats)
            st.header("Player Performance"); st.subheader("Batting Stats"); st.dataframe(batting_df.sort_values('runs', ascending=False))
            st.subheader("Bowling Stats"); st.dataframe(bowling_df.sort_values('wickets', ascending=False))

        with tab3:
            st.header("🏆 League & Season Analysis")
            leagues = league_df['league'].unique(); selected_league = st.selectbox("Select League", leagues)
            league_filtered = league_df[league_df['league'] == selected_league]
            seasons = league_filtered['season'].unique(); selected_season = st.selectbox("Select Season", seasons)
            
            season_data = league_filtered[league_filtered['season'] == selected_season]
            team_agg = season_data.groupby('team').agg(total_runs=('runs','sum'), total_wickets=('wickets','sum'), total_fours=('fours','sum'), total_sixes=('sixes','sum')).reset_index()
            
            st.subheader(f"Team Performance in {selected_league} ({selected_season})")
            st.dataframe(team_agg)
            
            st.subheader("Visualizations")
            chart_type = st.selectbox("Select Chart", ["Total Runs", "Total Wickets", "Total Fours", "Total Sixes"])
            if chart_type == "Total Runs": st.bar_chart(team_agg, x='team', y='total_runs')
            if chart_type == "Total Wickets": st.bar_chart(team_agg, x='team', y='total_wickets')
            if chart_type == "Total Fours": st.bar_chart(team_agg, x='team', y='total_fours')
            if chart_type == "Total Sixes": st.bar_chart(team_agg, x='team', y='total_sixes')

    else:
        st.warning("Could not process uploaded files.")
else:
    st.info("👋 Welcome! Upload JSON match logs to begin.")
