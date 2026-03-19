import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_data
from preprocess import preprocess_data
from features import engineer_features , select_features_for_model

df = load_data("data/raw/matches_1930_2022.csv")
df = preprocess_data(df)
df = engineer_features(df)
df = select_features_for_model(df)



# Sort by date (chronological)
df = df.sort_values('Date')

# Time-based split
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df.drop(columns=['match_result', 'Date'])
y_train = train_df['match_result']

X_test = test_df.drop(columns=['match_result', 'Date'])
y_test = test_df['match_result']

print(f"Train set: {len(X_train)} matches")
print(f"Test set: {len(X_test)} matches")


# Train XGBoost
model = XGBClassifier(
    n_estimators=100,      # Original
    max_depth=5,           # Original
    learning_rate=0.1,     # Original
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Draw', 'Home Win', 'Away Win']))

# Feature importance
feature_importance = pd.Series(
    model.feature_importances_, 
    index=X_train.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to models/model.pkl")

# Also save the feature names (you'll need this for predictions)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

print("Feature names saved to models/feature_names.pkl")





