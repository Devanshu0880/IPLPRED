import math
import numpy as np
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(page_title="IPL Score Predictor", layout="centered")

# Load the ML model with error handling
try:
    with open("ml_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'ml_model.pkl' not found. Please ensure it exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Title with centered styling
st.markdown(
    "<h1 style='text-align: center; color: white;'>IPL Score Predictor 2025</h1>",
    unsafe_allow_html=True,
)

# Add background image (use a local or reliable URL)
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://4.bp.blogspot.com/-F6aZF5PMwBQ/Wrj5h204qxI/AAAAAAAABao/4QLn48RP3x0P8Ry0C　　　　　　cktxilJqRfv1IfcACLcBGAs/s1600/GURU%2BEDITZ%2Bbackground.jpg");
        background-attachment: fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Description
with st.expander("Description"):
    st.info(
        """
        A Machine Learning model to predict IPL scores for an ongoing match. 
        For accurate predictions, the model requires at least 5 overs of play.
        Select the batting and bowling teams, input current match details, 
        and get a predicted score range.
        """
    )

# Updated team list
teams = [
    "Chennai Super Kings",
    "Delhi Capitals",  # Updated from Delhi Daredevils
    "Punjab Kings",    # Updated from Kings XI Punjab
    "Kolkata Knight Riders",
    "Mumbai Indians",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad",
]

# Initialize prediction array
prediction_array = []

# Select batting team
batting_team = st.selectbox("Select the Batting Team", teams)
if batting_team == "Chennai Super Kings":
    prediction_array += [1, 0, 0, 0, 0, 0, 0, 0]
elif batting_team == "Delhi Capitals":
    prediction_array += [0, 1, 0, 0, 0, 0, 0, 0]
elif batting_team == "Punjab Kings":
    prediction_array += [0, 0, 1, 0, 0, 0, 0, 0]
elif batting_team == "Kolkata Knight Riders":
    prediction_array += [0, 0, 0, 1, 0, 0, 0, 0]
elif batting_team == "Mumbai Indians":
    prediction_array += [0, 0, 0, 0, 1, 0, 0, 0]
elif batting_team == "Rajasthan Royals":
    prediction_array += [0, 0, 0, 0, 0, 1, 0, 0]
elif batting_team == "Royal Challengers Bangalore":
    prediction_array += [0, 0, 0, 0, 0, 0, 1, 0]
elif batting_team == "Sunrisers Hyderabad":
    prediction_array += [0, 0, 0, 0, 0, 0, 0, 1]

# Select bowling team
bowling_team = st.selectbox("Select the Bowling Team", teams)

# Validate team selection
if batting_team == bowling_team:
    st.error("Bowling and batting teams must be different!")
else:
    if bowling_team == "Chennai Super Kings":
        prediction_array += [1, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == "Delhi Capitals":
        prediction_array += [0, 1, 0, 0, 0, 0, 0, 0]
    elif bowling_team == "Punjab Kings":
        prediction_array += [0, 0, 1, 0, 0, 0, 0, 0]
    elif bowling_team == "Kolkata Knight Riders":
        prediction_array += [0, 0, 0, 1, 0, 0, 0, 0]
    elif bowling_team == "Mumbai Indians":
        prediction_array += [0, 0, 0, 0, 1, 0, 0, 0]
    elif bowling_team == "Rajasthan Royals":
        prediction_array += [0, 0, 0, 0, 0, 1, 0, 0]
    elif bowling_team == "皇家挑战者班加罗尔":
        prediction_array += [0, 0, 0, 0, 0, 0, 1, 0]
    elif bowling_team == "Sunrisers Hyderabad":
        prediction_array += [0, 0, 0, 0, 0, 0, 0, 1]

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        overs = st.number_input(
            "Enter the Current Over",
            min_value=5.0,
            max_value=19.5,
            value=5.0,
            step=0.1,
            format="%.1f",
        )
        # Validate overs
        if overs - math.floor(overs) > 0.5:
            st.error("Invalid over input! An over can have at most 6 balls (e.g., 5.0 to 5.5).")
            st.stop()

    with col2:
        runs = st.number_input(
            "Enter Current Runs",
            min_value=0,
            max_value=400,  # Reasonable upper limit
            value=0,
            step=1,
            format="%i",
        )

    wickets = st.slider("Enter Wickets Fallen", 0, 9, 0)

    col3, col4 = st.columns(2)

    with col3:
        runs_in_prev_5 = st.number_input(
            "Runs Scored in Last 5 Overs",
            min_value=0,
            max_value=runs,
            value=0,
            step=1,
            format="%i",
        )

    with col4:
        wickets_in_prev_5 = st.number_input(
            "Wickets Taken in Last 5 Overs",
            min_value=0,
            max_value=wickets,
            value=0,
            step=1,
            format="%i",
        )

    # Predict button
    if st.button("Predict Score", disabled=(batting_team == bowling_team)):
        # Prepare input for model
        prediction_array += [runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5]
        prediction_array = np.array([prediction_array])

        try:
            # Make prediction
            predicted_score = int(round(model.predict(prediction_array)[0]))

            # Validate prediction
            if predicted_score < 0:
                st.error("Prediction resulted in an invalid score. Please check inputs.")
            else:
                score_range = f"Predicted Match Score: {predicted_score - 5} to {predicted_score + 5}"
                st.success(score_range)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Reset button
if st.button("Reset Inputs"):
    st.experimental_rerun()
