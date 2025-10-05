import streamlit as st
import pandas as pd
import json
import os
import random
import time

# --- Core Data Processing Functions ---

def parse_match_data(uploaded_file):
    """
    Parses a single uploaded JSON cricket match file to extract key innings data.
    Now works with Streamlit's UploadedFile object.
    """
    try:
        # To read the uploaded file, we use getvalue() if it's an in-memory object
        data = json.load(uploaded_file)
    except (json.JSONDecodeError, AttributeError) as e:
        st.error(f"Error processing file {uploaded_file.name}: {e}")
        return []

    match_info = data.get('info', {})
    innings_data = data.get('innings', [])
    match_date = match_info.get('dates', [None])[0]
    venue = match_info.get('venue', 'N/A')
    
    parsed_innings = []

    for inning in innings_data:
        team = inning.get('team')
        total_runs = 0
        total_wickets = 0
        total_balls = 0

        overs = inning.get('overs', [])
        for over in overs:
            for delivery in over.get('deliveries', []):
                total_runs += delivery['runs']['total']
                # Count valid balls for over calculation
                if 'wides' not in delivery.get('extras', {}) and 'noballs' not in delivery.get('extras', {}):
                    total_balls += 1
                if 'wickets' in delivery:
                    total_wickets += len(delivery['wickets'])
        
        overs_bowled = total_balls / 6.0
        
        run_rate = (total_runs / overs_bowled) if overs_bowled > 0 else 0
        wicket_rate = (total_wickets / overs_bowled) if overs_bowled > 0 else 0

        parsed_innings.append({
            'match_date': match_date,
            'venue': venue,
            'team': team,
            'total_runs': total_runs,
            'wickets_lost': total_wickets,
            'overs_bowled': round(overs_bowled, 2),
            'run_rate': round(run_rate, 2),
            'wicket_rate': round(wicket_rate, 2)
        })

    return parsed_innings

def process_all_matches(uploaded_files):
    """
    Processes all uploaded JSON files to create a complete DataFrame.
    """
    all_match_data = []
    for uploaded_file in uploaded_files:
        all_match_data.extend(parse_match_data(uploaded_file))
    
    if not all_match_data:
        return pd.DataFrame()
        
    return pd.DataFrame(all_match_data)

# --- Simulation Function ---

def simulate_innings(team_name, avg_run_rate, avg_wicket_rate, overs=20):
    """
    Simulates a single T20 innings based on average rates.
    """
    total_score = 0
    wickets_fallen = 0
    simulation_log = []
    
    for over in range(1, overs + 1):
        if wickets_fallen >= 10:
            simulation_log.append(f"End of Over {over-1}: All out!")
            break

        # Simulate runs in the over with some variance around the average
        # Using a normal distribution; std deviation is set arbitrarily to 2.5
        runs_in_over = max(0, int(random.normalvariate(avg_run_rate, 2.5)))
        total_score += runs_in_over
        
        # Probabilistically determine if a wicket has fallen
        wickets_in_over = 0
        if random.random() < avg_wicket_rate:
            wickets_in_over = 1
            wickets_fallen += 1
        
        log_entry = f"**Over {over}:** {runs_in_over} runs"
        if wickets_in_over > 0 and wickets_fallen <= 10:
            log_entry += f", **1 WICKET!**"
        
        log_entry += f" -- **{team_name} {total_score}/{wickets_fallen}**"
        simulation_log.append(log_entry)
        
    return total_score, wickets_fallen, simulation_log


# --- Streamlit UI ---

st.set_page_config(page_title="Cricket Predictive Model", layout="wide")

st.title("🏏 Cricket Predictive Model")
st.markdown("By **Virat** for **bet365**")

# --- Sidebar for File Upload ---
st.sidebar.header("Upload Match Data")
uploaded_files = st.sidebar.file_uploader(
    "Upload JSON match logs",
    type=['json'],
    accept_multiple_files=True
)

# Load default file if none are uploaded
if not uploaded_files and os.path.exists('335982.json'):
    st.sidebar.info("Showing data from the default example file (`335982.json`). Upload your own files to analyze them.")
    with open('335982.json', 'r') as f:
        # To make it compatible with the function, we need to wrap it
        from io import StringIO
        # The file needs a 'name' attribute for the parsing function
        string_io_wrapper = StringIO(f.read())
        string_io_wrapper.name = '335982.json'
        uploaded_files = [string_io_wrapper]

# --- Main App Logic ---
if uploaded_files:
    with st.spinner('Analyzing match data...'):
        df = process_all_matches(uploaded_files)
    
    if not df.empty:
        # Calculate team averages for the simulator
        team_stats = df.groupby('team').agg({
            'run_rate': 'mean',
            'wicket_rate': 'mean'
        }).reset_index()
        team_stats['run_rate'] = team_stats['run_rate'].round(2)
        team_stats['wicket_rate'] = team_stats['wicket_rate'].round(2)
        
        # --- Create Tabs ---
        tab1, tab2 = st.tabs(["📊 Data Analysis", "⚔️ Match Simulator"])

        with tab1:
            st.header("Match Data Analysis")
            st.dataframe(df)

            st.header("Team Performance Summary")
            st.dataframe(team_stats)
            
            st.header("Average Run Rate by Team")
            # Ensure team names are strings for charting
            team_stats_chart = team_stats.set_index('team')
            st.bar_chart(team_stats_chart['run_rate'])

        with tab2:
            st.header("T20 Match Simulator")
            
            sim_mode = st.radio(
                "Choose Simulation Mode",
                ("Select Teams From Data", "Manual Input"),
                horizontal=True
            )
            
            st.markdown("---")

            col1, col2 = st.columns(2)
            
            team_a_name, team_a_rr, team_a_wr = "Team A", 8.5, 0.65
            team_b_name, team_b_rr, team_b_wr = "Team B", 8.2, 0.70

            unique_teams = team_stats['team'].unique()

            with col1:
                st.subheader("Team 1 (Batting First)")
                if sim_mode == "Select Teams From Data":
                    team_a_selection = st.selectbox("Select Team 1", unique_teams, key="team_a_select")
                    if team_a_selection:
                        stats = team_stats[team_stats['team'] == team_a_selection].iloc[0]
                        team_a_name = stats['team']
                        team_a_rr = stats['run_rate']
                        team_a_wr = stats['wicket_rate']
                        st.metric(label="Avg Run Rate", value=team_a_rr)
                        st.metric(label="Avg Wicket Rate (per over)", value=team_a_wr)
                else:
                    team_a_name = st.text_input("Team 1 Name", value=team_a_name)
                    team_a_rr = st.number_input("Avg Run Rate", value=team_a_rr, step=0.1, format="%.2f", key="team_a_rr")
                    team_a_wr = st.number_input("Avg Wicket Rate (per over)", value=team_a_wr, step=0.05, format="%.2f", key="team_a_wr")

            with col2:
                st.subheader("Team 2 (Chasing)")
                if sim_mode == "Select Teams From Data":
                    # Default to the second team in the list if available
                    default_index_b = 1 if len(unique_teams) > 1 else 0
                    team_b_selection = st.selectbox("Select Team 2", unique_teams, index=default_index_b, key="team_b_select")
                    if team_b_selection:
                        stats = team_stats[team_stats['team'] == team_b_selection].iloc[0]
                        team_b_name = stats['team']
                        team_b_rr = stats['run_rate']
                        team_b_wr = stats['wicket_rate']
                        st.metric(label="Avg Run Rate", value=team_b_rr)
                        st.metric(label="Avg Wicket Rate (per over)", value=team_b_wr)
                else:
                    team_b_name = st.text_input("Team 2 Name", value=team_b_name)
                    team_b_rr = st.number_input("Avg Run Rate", value=team_b_rr, step=0.1, format="%.2f", key="team_b_rr")
                    team_b_wr = st.number_input("Avg Wicket Rate (per over)", value=team_b_wr, step=0.05, format="%.2f", key="team_b_wr")

            st.markdown("---")

            if st.button("▶️ Run Simulation", use_container_width=True):
                with st.spinner("Simulating Innings 1..."):
                    time.sleep(1) # For dramatic effect
                    score1, wickets1, log1 = simulate_innings(team_a_name, team_a_rr, team_a_wr)
                
                target = score1 + 1
                
                with st.spinner(f"Simulating Innings 2 (Target: {target})..."):
                    time.sleep(1)
                    score2, wickets2, log2 = simulate_innings(team_b_name, team_b_rr, team_b_wr)

                st.subheader("Match Result")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.success(f"**{team_a_name}:** {score1}/{wickets1}")
                with res_col2:
                    st.info(f"**{team_b_name}:** {score2}/{wickets2}")

                # Determine Winner
                st.markdown("---")
                if score2 >= target:
                    st.balloons()
                    st.header(f"🎉 {team_b_name} wins by {10 - wickets2} wickets! 🎉")
                else:
                    st.header(f"🎉 {team_a_name} wins by {target - score2 - 1} runs! 🎉")
                
                # Display logs
                st.markdown("---")
                with st.expander(f"Show Innings 1 Log ({team_a_name})"):
                    for entry in log1:
                        st.markdown(entry)
                
                with st.expander(f"Show Innings 2 Log ({team_b_name})"):
                    for entry in log2:
                        st.markdown(entry)

    else:
        st.warning("Could not process the uploaded files. Please check the file format.")
else:
    st.info("👋 Welcome! Please upload one or more JSON match logs from the sidebar to begin.")
