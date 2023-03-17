import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the datasets
regular_season_results = pd.read_csv('MRegularSeasonDetailedResults.csv')
tournament_results = pd.read_csv('MNCAATourneyDetailedResults.csv')

# Merge regular season and tournament results
all_game_results = pd.concat([regular_season_results, tournament_results], ignore_index=True)

# Feature engineering and dataset preparation
all_game_results['point_diff'] = all_game_results['WScore'] - all_game_results['LScore']
all_game_results['team1_shooting_percentage'] = all_game_results['WFGM'] / all_game_results['WFGA']
all_game_results['team2_shooting_percentage'] = all_game_results['LFGM'] / all_game_results['LFGA']
all_game_results['rebounds_diff'] = all_game_results['WOR'] + all_game_results['WDR'] - (all_game_results['LOR'] + all_game_results['LDR'])
all_game_results['turnovers_diff'] = all_game_results['WTO'] - all_game_results['LTO']

X = all_game_results[['point_diff', 'team1_shooting_percentage', 'team2_shooting_percentage', 'rebounds_diff', 'turnovers_diff']]
y = (all_game_results['WTeamID'] < all_game_results['LTeamID']).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

def predict_winner(team1_id, team2_id, input_data, model):
    prediction = model.predict(input_data)
    return team1_id if prediction == 1 else team2_id

def calculate_team_average_stats(team_id, all_game_results):
    team_games = all_game_results[(all_game_results['WTeamID'] == team_id) | (all_game_results['LTeamID'] == team_id)]
    
    team_stats = {
        'point_diff': [],
        'team_shooting_percentage': [],
        'rebounds_diff': [],
        'turnovers_diff': []
    }
    
    for index, row in team_games.iterrows():
        if row['WTeamID'] == team_id:
            team_stats['point_diff'].append(row['WScore'] - row['LScore'])
            team_stats['team_shooting_percentage'].append(row['WFGM'] / row['WFGA'])
            team_stats['rebounds_diff'].append(row['WOR'] + row['WDR'] - (row['LOR'] + row['LDR']))
            team_stats['turnovers_diff'].append(row['WTO'] - row['LTO'])
        else:
            team_stats['point_diff'].append(row['LScore'] - row['WScore'])
            team_stats['team_shooting_percentage'].append(row['LFGM'] / row['LFGA'])
            team_stats['rebounds_diff'].append(row['LOR'] + row['LDR'] - (row['WOR'] + row['WDR']))
            team_stats['turnovers_diff'].append(row['LTO'] - row['WTO'])
    
    average_stats = {
        key: sum(values) / len(values)
        for key, values in team_stats.items()
    }
    return average_stats

def predict_game(team1_id, team2_id, model, all_game_results):
    team1_average_stats = calculate_team_average_stats(team1_id, all_game_results)
    team2_average_stats = calculate_team_average_stats(team2_id, all_game_results)

    input_data = pd.DataFrame([{
        'point_diff': team1_average_stats['point_diff'] - team2_average_stats['point_diff'],
        'team1_shooting_percentage': team1_average_stats['team_shooting_percentage'],
        'team2_shooting_percentage': team2_average_stats['team_shooting_percentage'],
        'rebounds_diff': team1_average_stats['rebounds_diff'] - team2_average_stats['rebounds_diff'],
        'turnovers_diff': team1_average_stats['turnovers_diff'] - team2_average_stats['turnovers_diff']
    }])

    winner = predict_winner(team1_id, team2_id, input_data, model)
    return winner

# Main loop for user input
while True:
    print("Enter the team IDs for the two teams you want to predict (e.g. 1101 1102) or type 'exit' to quit:")
    user_input = input()

    if user_input.lower() == 'exit':
        break

    try:
        team1_id, team2_id = map(int, user_input.split())
    except ValueError:
        print("Invalid input. Please enter two team IDs separated by a space.")
        continue

    winner = predict_game(team1_id, team2_id, model, all_game_results)
    print(f'The predicted winner is: {winner}')
