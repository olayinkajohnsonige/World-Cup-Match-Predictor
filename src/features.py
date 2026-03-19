def engineer_features(df):
    df = df.copy()

    # --------------------------------------------------
    # 1. TEAM WIN PERCENTAGE (Home)
    # --------------------------------------------------
    home_team_stats = df.groupby('home_team').agg({
        'match_result': [lambda x: (x == 1).sum(), 'count']
    }).reset_index()

    home_team_stats.columns = ['team', 'home_wins', 'home_matches']
    home_team_stats['home_win_pct'] = (
        home_team_stats['home_wins'] / home_team_stats['home_matches'] * 100
    ).round(2)

    home_win_dict = dict(zip(home_team_stats['team'], home_team_stats['home_win_pct']))

    # --------------------------------------------------
    # 2. TEAM WIN PERCENTAGE (Away)
    # --------------------------------------------------
    away_team_stats = df.groupby('away_team').agg({
        'match_result': [lambda x: (x == 2).sum(), 'count']
    }).reset_index()

    away_team_stats.columns = ['team', 'away_wins', 'away_matches']
    away_team_stats['away_win_pct'] = (
        away_team_stats['away_wins'] / away_team_stats['away_matches'] * 100
    ).round(2)

    away_win_dict = dict(zip(away_team_stats['team'], away_team_stats['away_win_pct']))

    # --------------------------------------------------
    # 3. MAP WIN %
    # --------------------------------------------------
    df['home_team_home_win_pct'] = df['home_team'].map(home_win_dict).fillna(0)
    df['away_team_away_win_pct'] = df['away_team'].map(away_win_dict).fillna(0)

    # --------------------------------------------------
    # 4. HOME ADVANTAGE
    # --------------------------------------------------
    df['home_advantage'] = df.apply(
        lambda row: 1 if row['home_team'] in row['Host'] else 0,
        axis=1
    )

    # --------------------------------------------------
    # 5. ROUND IMPORTANCE
    # --------------------------------------------------
    round_order = {
        'Group stage': 0,
        'First group stage': 0,
        'Second group stage': 1,
        'Group stage play-off': 1,
        'First round': 1,
        'Second round': 2,
        'Round of 16': 2,
        'Quarter-finals': 3,
        'Quarterfinals': 3,
        'Semi-finals': 4,
        'Semifinals': 4,
        'Third-place match': 4,
        'Third Place': 4,
        'Final stage': 5,
        'Final': 5
    }

    df['round_importance'] = df['Round'].map(round_order)

    # --------------------------------------------------
    # 6. RECENT FORM (OPTIMIZED)
    # --------------------------------------------------
    df = df.sort_values('Year')

    # Home recent form
    home_recent_form = []
    home_history = {}

    for _, row in df.iterrows():
        team = row['home_team']

        if team not in home_history:
            home_history[team] = []

        last_3 = home_history[team][-3:]
        form = sum(last_3) / len(last_3) if len(last_3) > 0 else 0
        home_recent_form.append(form)

        win = 1 if row['match_result'] == 1 else 0
        home_history[team].append(win * 100)

    df['home_team_recent_form'] = home_recent_form

    # Away recent form
    away_recent_form = []
    away_history = {}

    for _, row in df.iterrows():
        team = row['away_team']

        if team not in away_history:
            away_history[team] = []

        last_3 = away_history[team][-3:]
        form = sum(last_3) / len(last_3) if len(last_3) > 0 else 0
        away_recent_form.append(form)

        win = 1 if row['match_result'] == 2 else 0
        away_history[team].append(win * 100)

    df['away_team_recent_form'] = away_recent_form

    # --------------------------------------------------
    # 7. HEAD-TO-HEAD
    # --------------------------------------------------
    home_h2h = []
    away_h2h = []

    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        current_year = row['Year']

        h2h_matches = df[
            ((df['home_team'] == home_team) &
             (df['away_team'] == away_team)) &
            (df['Year'] < current_year)
        ]

        if len(h2h_matches) > 0:
            home_wins = (h2h_matches['match_result'] == 1).sum()
            away_wins = (h2h_matches['match_result'] == 2).sum()

            total = len(h2h_matches)

            home_h2h.append((home_wins / total) * 100)
            away_h2h.append((away_wins / total) * 100)
        else:
            home_h2h.append(0)
            away_h2h.append(0)

    df['home_h2h_record'] = home_h2h
    df['away_h2h_record'] = away_h2h

    # --------------------------------------------------
    # 8. EXPERIENCE
    # --------------------------------------------------
    df['home_experience'] = df.groupby('home_team').cumcount()
    df['away_experience'] = df.groupby('away_team').cumcount()

    # --------------------------------------------------
    # 9. DIFFERENCE FEATURES
    # --------------------------------------------------
    df['win_pct_diff'] = df['home_team_home_win_pct'] - df['away_team_away_win_pct']
    df['form_diff'] = df['home_team_recent_form'] - df['away_team_recent_form']
    df['experience_diff'] = df['home_experience'] - df['away_experience']

    
    # 10. DRAW TENDENCY (How often teams draw)
# --------------------------------------------------
    home_draw_pct_dict = df.groupby('home_team').apply(
    lambda x: (x['match_result'] == 0).sum() / len(x) * 100
).to_dict()

    away_draw_pct_dict = df.groupby('away_team').apply(
    lambda x: (x['match_result'] == 0).sum() / len(x) * 100
).to_dict()

    df['home_draw_tendency'] = df['home_team'].map(home_draw_pct_dict).fillna(0)
    df['away_draw_tendency'] = df['away_team'].map(away_draw_pct_dict).fillna(0)

    
    return df


def select_features_for_model(df):
    df = df.copy()

    features_to_keep = [
        'Date',
        'win_pct_diff',
        'form_diff',
        'experience_diff',
        'home_team_home_win_pct',
        'away_team_away_win_pct',
        'round_importance',
        'home_team_recent_form',
        'away_team_recent_form',
        'home_experience',
        'away_experience',
        'Year',
        'match_result'
    ]

    return df[features_to_keep]