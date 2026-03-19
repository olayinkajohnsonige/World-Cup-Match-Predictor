import pickle
import pandas as pd
import numpy as np
from src.preprocess import preprocess_data
from src.features import engineer_features, select_features_for_model

# Load the saved model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names
with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

def predict_match(home_team, away_team, year, round_name, host_country):
    """
    Predict a football match outcome
    """
    
    # Create minimal input with only necessary columns
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