# ⚽ World Cup Match Predictor

A machine learning-powered application that predicts World Cup football match outcomes using historical data and advanced feature engineering.

## 📊 Project Overview

This project builds an end-to-end machine learning pipeline to predict World Cup match results (Home Win, Away Win, or Draw) with 59% accuracy using XGBoost classification.

### Key Features
- **Data Processing**: Clean and preprocess 946 historical World Cup matches (1930-2022)
- **Feature Engineering**: 11 engineered features including team strength, experience, and tournament context
- **Machine Learning**: XGBoost classifier trained on time-based split to prevent data leakage
- **Interactive Dashboard**: Streamlit web app for real-time predictions
- **Model Performance**: 59% accuracy with strong home win prediction (75% recall)

## 🎯 Project Goals

1. Build a robust ML pipeline for sports prediction
2. Engineer meaningful features from historical data
3. Create an interactive user-friendly dashboard
4. Deploy a complete end-to-end project

## 📁 Project Structure
```
World-Cup-Match-Predictor/
├── data/
│   └── processed/
│   └── raw/
│       └── matches_1930_2022.csv    # Raw World Cup data
├── models/
│   ├── model.pkl                         # Trained XGBoost model
│   └── feature_names.pkl                 # Feature ordering for predictions
├── notebooks/                            # Jupyter notebooks for exploration
├── scripts/
│   └── cleaned_data_sa...                # Data cleaning scripts
├── src/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── data_loader.py                    # Load and prepare data
│   ├── preprocess.py                     # Data cleaning and preprocessing
│   ├── features.py                       # Feature engineering logic
│   └── train.py                          # Model training  evaluation
├── app.py                                # Streamlit dashboard
├── predict.py                            # Prediction interface
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
└── .gitignore
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/olayinkajohnsonige/World-Cup-Match-Predictor
cd World-Cup-Match-Predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard**
```bash
streamlit run app.py
```

4. **Open in browser**
- Navigate to `http://localhost:8501`

## 📊 How to Use the Dashboard

1. **Fill in match details** in the left sidebar:
   - Home Team (e.g., "Argentina")
   - Away Team (e.g., "France")
   - Year (1930-2030)
   - Tournament Round (Group stage, Quarter-finals, Final, etc.)
   - Host Country (e.g., "USA")

2. **Click "Predict Match"**

3. **View results**:
   - Probability breakdown for each outcome (Draw, Home Win, Away Win)
   - Predicted outcome with confidence score
   - Interactive bar chart visualization
   - Match details summary

## 🤖 Model Details

### Algorithm & Architecture
- **Model**: XGBoost Classifier
- **Hyperparameters**:
  - n_estimators: 100
  - max_depth: 5
  - learning_rate: 0.1
  - eval_metric: mlogloss

### Training Data
- **Dataset**: 946 World Cup matches (1930-2022)
- **Train/Test Split**: 80/20 (time-based, chronological)
- **Features**: 11 engineered features

### Model Performance
```
Overall Accuracy: 59%

Per-Class Performance:
  Draw:     Precision: 0.35 | Recall: 0.32 | F1: 0.33
  Home Win: Precision: 0.70 | Recall: 0.75 | F1: 0.72
  Away Win: Precision: 0.57 | Recall: 0.55 | F1: 0.56
```

### Top 5 Feature Importance
1. **win_pct_diff** (16.4%) - Win percentage difference between teams
2. **Year** (10.7%) - Time period/era of the match
3. **away_experience** (10.2%) - Away team's total match count
4. **away_team_away_win_pct** (9.0%) - Away team's win rate in away matches
5. **away_team_recent_form** (8.7%) - Away team's performance in last 3 tournaments

## 🔧 Feature Engineering

The model uses 11 engineered features:

| Feature | Description | Importance |
|---------|-------------|-----------|
| `win_pct_diff` | Difference in home/away win percentages | 16.4% |
| `form_diff` | Difference in recent tournament performance | 6.2% |
| `experience_diff` | Difference in match count experience | 8.1% |
| `home_team_home_win_pct` | Home team's win rate at home | 8.4% |
| `away_team_away_win_pct` | Away team's win rate away | 9.0% |
| `round_importance` | Tournament stage (0=group, 5=final) | 8.2% |
| `home_team_recent_form` | Home team's last 3 tournaments avg | 7.5% |
| `away_team_recent_form` | Away team's last 3 tournaments avg | 8.7% |
| `home_experience` | Home team's total matches played | 7.5% |
| `away_experience` | Away team's total matches played | 10.2% |
| `Year` | World Cup year | 10.7% |

## 📈 Model Development Process

### 1. Data Preprocessing (`src/preprocess.py`)
- Loaded 946 World Cup matches from 1930-2022
- Dropped columns causing data leakage (scores, detailed goal data)
- Dropped irrelevant columns (captains, managers, referees)
- Handled missing values
- Created target variable: 0=Draw, 1=Home Win, 2=Away Win
- Converted Date to datetime

### 2. Feature Engineering (`src/features.py`)
- **Team Win Percentage**: Home and away win rates for each team
- **Home Advantage**: Binary flag if team playing in home country
- **Round Importance**: Ordinal encoding of tournament stage
- **Recent Form**: Average win % in last 3 World Cup tournaments
- **Head-to-Head**: Historical win % between specific team pairs
- **Experience**: Total matches played by each team
- **Difference Features**: Direct comparison of team strengths

### 3. Model Training (`src/train.py`)
- Compared Random Forest (56% accuracy) vs XGBoost (59% accuracy)
- Used time-based train/test split to prevent data leakage
- 80% training (771 matches), 20% testing (193 matches)
- Hyperparameter tuning for optimal performance

### 4. Dashboard (`app.py`)
- Built interactive Streamlit interface
- Real-time predictions with probability visualization
- Plotly charts for probability breakdown

## 🐛 Known Limitations

- **Draws are hard to predict** (32% recall) due to class imbalance in data (only 22% of matches)
- **Model uses generic team statistics** - predictions don't account for match-specific factors like injuries or tactical changes
- **Football is inherently random** - 59% accuracy is reasonable but far from perfect
- **Limited by historical data** - no player stats, possession data, or tactical information available
- **Temporal patterns** - Year feature helps but doesn't capture modern tactical evolution

## 🔮 Future Improvements

- [ ] Implement rolling window features to properly handle temporal data
- [ ] Add player-level statistics (if data available)
- [ ] Include tactical formation and possession data
- [ ] Try neural networks for comparison
- [ ] Deploy to Streamlit Cloud for public access
- [ ] Build historical match database for reference
- [ ] Improve draw prediction with SMOTE or other class balancing
- [ ] Add confidence calibration analysis
- [ ] Create team comparison visualizations

## 🛠️ Technologies Used

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: XGBoost, scikit-learn
- **Visualization**: Plotly, Streamlit
- **Data Loading**: Custom data_loader.py
- **Version Control**: Git, GitHub

## 📚 Key Files

| File | Purpose |
|------|---------|
| `src/data_loader.py` | Load CSV data |
| `src/preprocess.py` | Clean and validate data |
| `src/features.py` | Engineer features from raw data |
| `src/train.py` | Train and evaluate XGBoost model |
| `predict.py` | Load model and make predictions |
| `app.py` | Streamlit dashboard interface |

## 📖 Learning Outcomes

This project demonstrates:
- ✅ Complete ML pipeline: data → preprocessing → features → model → deployment
- ✅ Feature engineering for structured data
- ✅ Time-series evaluation (avoiding data leakage)
- ✅ Model comparison and optimization
- ✅ Interactive dashboard creation with Streamlit
- ✅ Production-ready code organization
- ✅ Documentation and version control best practices

## 🤝 Contributing

Feedback and improvements are welcome! Feel free to:
- Report issues
- Suggest model improvements
- Fork and experiment with different features

## 📄 License

MIT License - feel free to use this project for learning purposes

## 👤 Author

Your Name - [Your GitHub](https://github.com/olayinkajohnsonige)

## 🙏 Acknowledgments

- World Cup historical data
- XGBoost and scikit-learn communities
- Streamlit framework for easy dashboard creation

---



Last Updated: March 2026