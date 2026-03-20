import pickle
import pandas as pd
import numpy as np
from src.preprocess import preprocess_data
from src.features import engineer_features, select_features_for_model
from src.data_loader import load_data

# Load historical data once for team statistics
try:
    historical_data = load_data("data/raw/matches_1930_2022.csv")
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
        return None
    
    if is_home:
        team_data = historical_data[historical_data['home_team'] == team_name]
    else:
        team_data = historical_data[historical_data['away_team'] == team_name]
    
    if len(team_data) == 0:
        return None
    
    return team_data.iloc[-1]

def predict_match(home_team, away_team, year, round_name, host_country):
    """
    Predict a football match outcome using rule-based system
    Based on team strength comparison
    """
    
    print(f"\n🔍 Predicting: {home_team} vs {away_team}")
    
    # Get real team stats
    home_stats = get_team_stats(home_team, is_home=True)
    away_stats = get_team_stats(away_team, is_home=False)
    
    if home_stats is not None and away_stats is not None:
        print(f"✅ Using real team statistics")
        
        # Extract key metrics
        home_win_pct = home_stats['home_team_home_win_pct']
        away_win_pct = away_stats['away_team_away_win_pct']
        home_form = home_stats['home_team_recent_form']
        away_form = away_stats['away_team_recent_form']
        
        # Calculate strength difference
        win_pct_diff = home_win_pct - away_win_pct
        form_diff = home_form - away_form
        
        # Combined score
        home_strength = (win_pct_diff + form_diff) / 2
        
        print(f"Home: {home_team} (Win%: {home_win_pct:.1f}%, Form: {home_form:.1f}%)")
        print(f"Away: {away_team} (Win%: {away_win_pct:.1f}%, Form: {away_form:.1f}%)")
        print(f"Strength Difference: {home_strength:.1f}%")
        
        # Rule-based prediction
        if home_strength > 20:
            # Home team significantly stronger
            prediction = 'Home Win'
            confidence = min(85, 55 + (home_strength / 3))
            probabilities = {
                'Home Win': confidence,
                'Draw': 30 - (home_strength / 10),
                'Away Win': 40 - (home_strength / 5)
            }
        elif home_strength < -20:
            # Away team significantly stronger
            prediction = 'Away Win'
            confidence = min(85, 55 + (abs(home_strength) / 3))
            probabilities = {
                'Away Win': confidence,
                'Draw': 30 - (abs(home_strength) / 10),
                'Home Win': 40 - (abs(home_strength) / 5)
            }
        elif abs(home_strength) < 5:
            # Very close match - likely competitive or draw
            prediction = 'Draw'
            confidence = 50
            probabilities = {
                'Draw': confidence,
                'Home Win': 25,
                'Away Win': 25
            }
        else:
            # Slight advantage to one team
            if home_strength > 0:
                prediction = 'Home Win'
                confidence = 40 + home_strength
            else:
                prediction = 'Away Win'
                confidence = 40 + abs(home_strength)
            
            probabilities = {
                'Home Win': 45 + (home_strength / 2),
                'Draw': 40 - abs(home_strength / 2),
                'Away Win': 45 - (home_strength / 2)
            }
        
        # Normalize probabilities to sum to 100
        total = sum(probabilities.values())
        probabilities = {k: (v/total)*100 for k, v in probabilities.items()}
        
        return {
            'prediction': prediction,
            'home_team': home_team,
            'away_team': away_team,
            'confidence': float(confidence),
            'probabilities': {k: float(v) for k, v in probabilities.items()}
        }
    
    else:
        # Fallback - default prediction
        print(f"⚠️ Teams not found in historical data, using default prediction")
        return {
            'prediction': 'Draw',
            'home_team': home_team,
            'away_team': away_team,
            'confidence': 50.0,
            'probabilities': {
                'Draw': 50.0,
                'Home Win': 25.0,
                'Away Win': 25.0
            }
        }