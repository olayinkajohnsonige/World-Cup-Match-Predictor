import pandas as pd


def create_target(df):
    df["match_result"] = 0
    df.loc[df["home_score"] > df["away_score"], "match_result"] = 1
    df.loc[df["home_score"] < df["away_score"], "match_result"] = 2
    return df




def clean_data(df):
    df = df.copy()
    
    columns_to_drop = [
        "home_score", "away_score", "home_manager", "away_manager", "Score",
        "home_captain", "away_captain", "away_penalty", "home_penalty",
        "home_xg", "away_xg", "Notes", "Officials", "Referee",
        "home_goal", "away_goal", "home_goal_long", "away_goal_long",
        "home_own_goal", "away_own_goal", "home_penalty_goal", "away_penalty_goal",
        "home_penalty_miss_long", "away_penalty_miss_long",
        "home_penalty_shootout_goal_long", "away_penalty_shootout_goal_long",
        "home_penalty_shootout_miss_long", "away_penalty_shootout_miss_long",
        "home_red_card", "away_red_card", "home_yellow_card_long", "away_yellow_card_long",
        "home_yellow_red_card", "away_yellow_red_card",
        "home_substitute_in_long", "away_substitute_in_long"
    ]
    
    # Only drop columns that actually exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    df = df.drop(columns=columns_to_drop)
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df


def preprocess_data(df):
    df=df.copy()
    df=create_target(df)
    df=clean_data(df)
    return df

