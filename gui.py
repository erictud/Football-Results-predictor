import gradio as gr
import pandas as pd
import joblib

# loading model trained
model = joblib.load("football_predict.joblib")
print("Coloanele EXACTE cerute de model sunt:", model.feature_names_in_)

def predict_match(home_team, away_team, home_goal, away_goal):        
    # goals need to be ints
    host_ht_goals = int(home_goal)
    away_ht_goals = int(away_goal)
    
    # calculating data
    goal_diff_ht = host_ht_goals - away_ht_goals
    median_goals_per_min_ht = (host_ht_goals + away_ht_goals) / 45.0
    
    # calculating advantage team (0 tie, 1 home, 2 away)
    if host_ht_goals > away_ht_goals:
        advatage_ht = 1
    elif away_ht_goals > host_ht_goals:
        advatage_ht = 2
    else:
        advatage_ht = 0
        
    # creating input data frame
    input_data = pd.DataFrame([{
        "HT goals of home team": host_ht_goals,
        "HT goals away team": away_ht_goals,
        "Goal diff HT": goal_diff_ht,
        "HT advantage": advatage_ht,
        "Goals per minute at HT": median_goals_per_min_ht
    }])
    
    # getting prediciton
    prediction = model.predict(input_data)[0]
    
    # Showing output
    if prediction == 1:
        return f"🏆 Victory for {home_team} (Home)"
    elif prediction == 2:
        return f"🏆 Victory for {away_team} (Away)"
    else:
        return "⚖️ Tie"

# Constructing web interface
interface = gr.Interface(
    fn=predict_match, # function that runs when we submit
    inputs=[
        gr.Textbox(label="Home Team", value="FCSB", placeholder="Home team name"),
        gr.Textbox(label="Away Team", value="Dinamo", placeholder="Away team name"),
        gr.Number(label="Home Team Goal", value=0, precision=0),
        gr.Number(label="Away Team Goal", value=0, precision=0)
    ],
    outputs=gr.Textbox(label="Predicted result"),
    title="Enter match details for prediction",
    description="Prediciting the result of a match based on the result at half time",
)

# running interface
if __name__ == "__main__":
    print("Running GUI...")
    interface.launch(share=False)