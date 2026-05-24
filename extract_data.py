import pandas 
import requests
from dotenv import load_dotenv 
import os
from sklearn.model_selection import train_test_split

# Getting API key
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Request details
request_url = "https://v3.football.api-sports.io/fixtures"
request_headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}
seasons = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"] # extracting match data from 3 years

matches = []

for season in seasons:
    query_parameters = {"league": "283", "season": season} # 283 id of Superliga
    response = requests.get(request_url, headers=request_headers, params=query_parameters)
    respone_data = response.json()

    # data processing
    for match in respone_data["response"]:
        # checking if the match is over in order to extract de data
        if match["fixture"]["status"]["short"] == "FT": 

            # extracting data
            host_team = match["teams"]["home"]["name"]
            away_team = match["teams"]["away"]["name"]
            host_ht_goals = match["score"]["halftime"]["home"] # goals of the home team at  HT
            away_ht_goals = match["score"]["halftime"]["away"] # goals of the away team at HT
            host_ft_goals = match["goals"]["home"] # goals of the home team at the end
            away_ft_goals = match["goals"]["away"] # goals of thw away team at the end

            # derriving stats based on extracted data
            goal_diff_ht = host_ht_goals - away_ht_goals # diffrence between goals at HT
            goal_total_ht = host_ht_goals + away_ht_goals # total goals at HT
            median_goals_per_min_ht = round(goal_total_ht / 45.0, 2) # goals scored per minute in the first half
            advatage_ht = 0 # 0 if draw, 1 if home leads, 2 if away leads
            final_res = 0   # 0 in case of draw, 1 if home team wins, 2 if away wins

            if goal_diff_ht > 0:
                advatage_ht = 1
            elif goal_diff_ht < 0:
                advatage_ht = 2

            if host_ft_goals > away_ft_goals:
                final_res = 1
            elif away_ft_goals > host_ft_goals:
                final_res = 2
            
            # adding processed data
            matches.append([host_team, away_team, host_ht_goals, away_ht_goals, goal_diff_ht, goal_total_ht, advatage_ht, median_goals_per_min_ht, final_res])
            
# constructing data frame
columns = ["Home team", "Away team", "HT goals of home team", "HT goals away team", "Goal diff HT", "Total goals HT", "HT advantage", "Goals per minute at HT","Result"]

df_matches = pandas.DataFrame(matches, columns=columns)

# splitting dataframe in one for testing and one for training (70% training, 30% testing)
df_train, df_test = train_test_split(df_matches, test_size=0.3, random_state=42)

# Exporting dataframes into csv files
df_train.to_csv("train.csv", index=False) # dont want rows to have indexes
df_test.to_csv("test.csv", index=False)

# Printing succes msg
print(f"\033[92m Training and testing models have been succesfully exported into .csv files! \033[0m") # making output in green color