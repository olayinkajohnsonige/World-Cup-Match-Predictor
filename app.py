import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
sys.path.append('src')
from predict import predict_match

# Page config
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# Title
st.title("⚽ World Cup Match Predictor")
st.write("Predict the outcome of football matches using machine learning")

# Sidebar for inputs
st.sidebar.header("Match Details")

# Team selection
home_team = st.sidebar.text_input("Home Team", value="Argentina")
away_team = st.sidebar.text_input("Away Team", value="France")

# Year
year = st.sidebar.slider("Year", 1930, 2030, 2026)

# Round
round_name = st.sidebar.selectbox(
    "Tournament Round",
    ["Group stage", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
)

# Host country
host_country = st.sidebar.text_input("Host Country", value="USA")

# Predict button
if st.sidebar.button("Predict Match", key="predict_button"):
    # Get prediction
    result = predict_match(home_team, away_team, year, round_name, host_country)
    
    # Display results
    st.header(f"{home_team} vs {away_team}")
    
    # Main prediction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Draw", f"{result['probabilities']['Draw']:.1f}%")
    
    with col2:
        st.metric("Home Win", f"{result['probabilities']['Home Win']:.1f}%")
    
    with col3:
        st.metric("Away Win", f"{result['probabilities']['Away Win']:.1f}%")
    
    # Prediction
    st.success(f"**Predicted Outcome: {result['prediction']}** (Confidence: {result['confidence']:.1f}%)")
    
    # Chart
    fig = go.Figure(data=[
        go.Bar(
            x=['Draw', 'Home Win', 'Away Win'],
            y=[
                result['probabilities']['Draw'],
                result['probabilities']['Home Win'],
                result['probabilities']['Away Win']
            ],
            marker=dict(color=['#FFA500', '#00AA00', '#0000FF'])
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Outcome",
        yaxis_title="Probability (%)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Details
    st.subheader("Prediction Details")
    st.write(f"- **Year:** {year}")
    st.write(f"- **Round:** {round_name}")
    st.write(f"- **Host Country:** {host_country}")

else:
    st.info("👈 Fill in the match details and click 'Predict Match' to see results")