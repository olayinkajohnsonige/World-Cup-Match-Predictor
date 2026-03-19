import pickle
import pandas as pd
import numpy as np
from src.preprocess import preprocess_data
from src.features import engineer_features, select_features_for_model
from src.data_loader import load_data

# Load the saved model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names
with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load historical data once for team statistics
try:
    historical_data = load_data("data/processed/raw/matches_1930_2022.csv")
    print(f"✅ Loaded {len(historical_data)} historical matches")
    historical_data = preprocess_data(historical_data)
    historical_data = engineer_features(historical_data)
    print(f"✅ Features engineered")
except Exception as e:
    print(f"❌ Error loading historical data: {e}")
    historical_data = None

def get_team_stats(team_name, is_home=True):
    """Get historical stats for a team from the dataset"""
    if historical_data is None:
        print(f"❌ No historical data available")
        return None
    
    if is_home:
        team_data = historical_data[historical_data['home_team'] == team_name]
    else:
        team_data = historical_data[historical_data['away_team'] == team_name]
    
    if len(team_data) == 0:
        print(f"❌ Team '{team_name}' not found in historical data")
        return None
    
    print(f"✅ Found {len(team_data)} matches for {team_name}")
    return team_data.iloc[-1]  # Get most recent record

def predict_match(home_team, away_team, year, round_name, host_country):
    """
    Predict a football match outcome using real team statistics
    """
    
    print(f"\n🔍 Predicting: {home_team} vs {away_team}")
    
    # Get real team stats from historical data
    home_stats = get_team_stats(home_team, is_home=True)
    away_stats = get_team_stats(away_team, is_home=False)
    
    # If both teams found in historical data, use real stats
    if home_stats is not None and away_stats is not None:
        print(f"✅ Using real team statistics")
        
        # Map round name to importance
        round_importance_map = {
            'Group stage': 0,
            'First group stage': 0,
            'Second group stage': 1,
            'Round of 16': 2,
            'Quarter-finals': 3,
            'Semi-finals': 4,
            'Final': 5
        }
        round_imp = round_importance_map.get(round_name, 2)
        
        # Create input with real team statistics
        input_data = pd.DataFrame({
            'Date': [home_stats['Date']],
            'win_pct_diff': [home_stats['home_team_home_win_pct'] - away_stats['away_team_away_win_pct']],
            'form_diff': [home_stats['home_team_recent_form'] - away_stats['away_team_recent_form']],
            'experience_diff': [home_stats['home_experience'] - away_stats['away_experience']],
            'home_team_home_win_pct': [home_stats['home_team_home_win_pct']],
            'away_team_away_win_pct': [away_stats['away_team_away_win_pct']],
            'round_importance': [round_imp],
            'home_team_recent_form': [home_stats['home_team_recent_form']],
            'away_team_recent_form': [away_stats['away_team_recent_form']],
            'home_experience': [home_stats['home_experience']],
            'away_experience': [away_stats['away_experience']],
            'Year': [year],
            'match_result': [0]
        })
        
        print(f"Features: win_pct_diff={input_data['win_pct_diff'].values[0]:.2f}, form_diff={input_data['form_diff'].values[0]:.2f}")
    else:
        print(f"⚠️ Using fallback dummy data (teams not in historical dataset)")
        # Fallback: create dummy input if teams not found
        input_data = pd.DataFrame({
            'home_team': [home_team],
            'away_team': [away_team],
            'Year': [year],
            'Round': [round_name],
            'Host': [host_country],
            'home_score': [0],
            'away_score': [0],
            'Date': ['2026-01-01'],
            'Venue': ['Unknown'],
            'home_manager': ['Unknown'],
            'away_manager': ['Unknown'],
            'Attendance': [0],
            'Score': ['0-0'],
            'Notes': ['']
        })
        
        # Preprocess and engineer features
        input_data = preprocess_data(input_data)
        input_data = engineer_features(input_data)
        input_data = select_features_for_model(input_data)
    
    # Drop Date and match_result for prediction
    X = input_data.drop('match_result', axis=1).drop('Date', axis=1)
    
    # Ensure features are in correct order
    X = X[feature_names]
    
    # Make prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Map to readable names
    outcome_names = {0: 'Draw', 1: 'Home Win', 2: 'Away Win'}
    
    return {
        'prediction': outcome_names[prediction],
        'home_team': home_team,
        'away_team': away_team,
        'confidence': float(probabilities[prediction]) * 100,
        'probabilities': {
            'Draw': float(probabilities[0]) * 100,
            'Home Win': float(probabilities[1]) * 100,
            'Away Win': float(probabilities[2]) * 100
        }
    }