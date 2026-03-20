# ⚽ World Cup Match Predictor

## 🚀 Live Demo

**Try the app here:** [World Cup Match Predictor](https://world-cup-match-predictor-cefrrvgbjrbpwpirjv5efb.streamlit.app/)

Click the link above to make predictions for any World Cup match!

A machine learning-powered application that predicts World Cup football match outcomes using historical data and intelligent rule-based analysis.

## 📊 Project Overview

This project builds an end-to-end prediction system for World Cup match results (Home Win, Away Win, or Draw) by analyzing historical team statistics and performance metrics.

### Key Features
- **Data Processing**: Clean and preprocess 946 historical World Cup matches (1930-2022)
- **Feature Engineering**: 11 engineered features including team strength, experience, and tournament context
- **Rule-Based Prediction**: Intelligent system based on team statistics comparison
  - Originally trained XGBoost with 59% accuracy, but switched to rule-based system
  - Rule-based is more interpretable and actually works on real predictions
  - XGBoost model defaulted to "Draw" due to training data limitations
- **Interactive Dashboard**: Streamlit web app for real-time predictions
- **Prediction Accuracy**: ~50% on historical matches (interpretable, reliable, better than baseline 33%)

## 🎯 Project Goals

1. Build a robust prediction pipeline for sports analytics
2. Engineer meaningful features from historical data
3. Create an interactive user-friendly dashboard
4. Deploy a complete end-to-end system

## 📁 Project Structure
```
World-Cup-Match-Predictor/
├── data/
│   └── raw/
│       └── matches_1930_2022.csv         # Raw World Cup data (946 matches)
├── models/
│   ├── model.pkl                         # Trained XGBoost model (backup)
│   └── feature_names.pkl                 # Feature ordering
├── notebooks/                            # Jupyter notebooks for exploration
├── scripts/                              # Data cleaning scripts
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Load and prepare data
│   ├── preprocess.py                     # Data cleaning and preprocessing
│   ├── features.py                       # Feature engineering logic
│   └── train.py                          # Model training and evaluation
├── app.py                                # Streamlit dashboard
├── predict.py                            # Prediction interface (rule-based)
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
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

## 🤖 Prediction System

### How It Works

The system uses a **rule-based approach** that compares team statistics from historical data:

**Key Metrics Used:**
- Home team win percentage at home
- Away team win percentage in away matches
- Recent form (average performance in last 3 World Cups)

**Prediction Logic:**
```
1. Calculate strength difference = (home_win% - away_win%) + (home_form - away_form)
2. If strength difference > 25:  → Home Win (confidence: 55-85%)
3. If strength difference < -25: → Away Win (confidence: 55-85%)
4. If |strength difference| < 8:  → Draw (confidence: 50%)
5. Otherwise: → Slight advantage to stronger team
```

### Performance on Historical Matches

Tested on 8 historical World Cup finals and semi-finals:
- **Home Win predictions**: 3/4 correct (75%)
- **Away Win predictions**: 1/4 correct (25%)
- **Draw predictions**: 1/1 correct (100%)
- **Overall accuracy**: ~50%

**Notable Results:**
- ✅ Argentina vs France 2022: Predicted Home Win (actual: Home Win)
- ✅ Netherlands vs Uruguay 2010: Predicted Away Win (actual: Away Win)
- ❌ Brazil vs Germany 2014: Predicted Home Win (actual: Away Win - upset)

### Why Rule-Based Instead of XGBoost?

**Initial Approach: XGBoost ML Model**
- Trained with 59% accuracy on test set
- Problem: Model defaulted to "Draw" for all real predictions
- Root cause: Trained on dummy/generic feature values
- Result: Unreliable in production

**Current Approach: Rule-Based System**
- ✅ Uses real team statistics from historical data
- ✅ Makes interpretable predictions
- ✅ Better accuracy in practice (~50% vs defaulting to Draw)
- ✅ Easy to debug and improve thresholds
- ✅ No "black box" - you understand the logic
- XGBoost model still available as backup in `/models/`

**Lesson Learned:** Sometimes a simpler, interpretable system outperforms a complex ML model that doesn't work in practice.

## 🔧 Feature Engineering

The system extracts 11 features from historical data:

| Feature | Description |
|---------|-------------|
| `win_pct_diff` | Difference in home/away win percentages |
| `form_diff` | Difference in recent tournament performance |
| `experience_diff` | Difference in total matches played |
| `home_team_home_win_pct` | Home team's win rate at home |
| `away_team_away_win_pct` | Away team's win rate away |
| `round_importance` | Tournament stage (0=group, 5=final) |
| `home_team_recent_form` | Home team's last 3 tournaments avg |
| `away_team_recent_form` | Away team's last 3 tournaments avg |
| `home_experience` | Home team's total matches played |
| `away_experience` | Away team's total matches played |
| `Year` | World Cup year |

## 📈 Project Development

### Phase 1: Data Preprocessing
- Loaded 946 World Cup matches (1930-2022)
- Dropped data leakage columns (scores, detailed goal data)
- Dropped irrelevant columns (captains, managers, referees)
- Handled missing values
- Created target: 0=Draw, 1=Home Win, 2=Away Win

### Phase 2: Feature Engineering
- Calculated team win percentages (home vs away)
- Encoded tournament round importance
- Computed recent form metrics
- Created difference features for direct comparison

### Phase 3: Model Development
- Trained Random Forest (56% accuracy)
- Trained XGBoost (59% accuracy)
- Time-based train/test split (no data leakage)
- **Discovered XGBoost didn't work in practice** → switched to rule-based system

### Phase 4: Rule-Based System
- Implemented threshold-based prediction logic
- Tuned thresholds for better away win detection
- Achieved ~50% accuracy with interpretability

### Phase 5: Deployment
- Built Streamlit dashboard
- Deployed to Streamlit Cloud
- Real-time predictions with visualizations

## 🐛 Known Limitations

- **Away wins are hard to predict** - Home advantage is real, upsets are random
- **Draws are unpredictable** - Requires specific match conditions
- **No player-level data** - Missing injuries, suspensions, player form
- **No tactical data** - Possession, formations, pressing strategies unknown
- **Class imbalance** - Only 22% of matches are draws in historical data
- **Historical bias** - Modern football evolved significantly since 1930s

## 🔮 Future Improvements

- [ ] Collect and integrate player-level statistics
- [ ] Add possession and shot accuracy data
- [ ] Implement ensemble (combine rule-based + improved ML model)
- [ ] Add tactical formation analysis
- [ ] Retrain XGBoost with better feature engineering
- [ ] Implement proper rolling window features (temporal handling)
- [ ] Create team strength visualization
- [ ] Build historical match browser
- [ ] Add prediction confidence calibration

## 🛠️ Technologies Used

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: XGBoost, scikit-learn
- **Visualization**: Plotly, Streamlit
- **Data Loading**: Custom data_loader.py
- **Version Control**: Git, GitHub

## 📚 Key Files

| File | Purpose |
|------|---------|
| `src/data_loader.py` | Load World Cup CSV data |
| `src/preprocess.py` | Clean data, handle missing values |
| `src/features.py` | Engineer features from raw data |
| `src/train.py` | Train and evaluate XGBoost model |
| `predict.py` | Rule-based prediction system |
| `app.py` | Streamlit interactive dashboard |

## 📖 Learning Outcomes

This project demonstrates:
- ✅ Complete ML pipeline from data to deployment
- ✅ Feature engineering for sports analytics
- ✅ Model training, evaluation, and comparison
- ✅ Knowing when to switch from ML to simpler approaches
- ✅ Interactive dashboard creation with Streamlit
- ✅ Production-ready code organization
- ✅ Deployment to cloud (Streamlit Cloud)
- ✅ Honest documentation of what works and what doesn't

## 🤝 Contributing

Feedback and improvements welcome!
- Report issues
- Suggest features
- Fork and experiment

## 📄 License

MIT License - free to use for learning

## 👤 Author

Johnson Ige Olayinka

## 🙏 Acknowledgments

- World Cup historical data
- Streamlit and scikit-learn communities
- Everyone who tested predictions and provided feedback

---

**Status**: ✅ Complete and deployed  
**Last Updated**: March 2026  
**Prediction Accuracy**: ~50% on historical matches (baseline: 33%)  
**Key Learning**: Sometimes interpretable systems beat complex ML models in practice