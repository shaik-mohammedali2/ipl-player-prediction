import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="IPL Live Match Prediction - Shaik Mohammed Ali", layout="centered")

st.title("🏏 IPL Live Match Prediction")
st.write("Control match situation and predict outcome.")

try:
    model = joblib.load("ipl_model.pkl")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# --------------------------
# TEAMS & CITY
# --------------------------
batting_team = st.selectbox("🏏 Select Batting Team", [
    "CSK", "MI", "RCB", "KKR", "SRH", "RR", "PBKS", "GT", "LSG", "DC"
])

bowling_team = st.selectbox("🎯 Select Bowling Team", [
    "CSK", "MI", "RCB", "KKR", "SRH", "RR", "PBKS", "GT", "LSG", "DC"
])

city = st.selectbox("📍 Host City", [
    "Mumbai", "Chennai", "Bangalore", "Kolkata", "Hyderabad",
    "Delhi", "Ahmedabad", "Jaipur"
])

# --------------------------
# MATCH SETTINGS
# --------------------------
target = st.number_input("🎯 Target Score", value=250.0)

st.subheader("🏏 Match Situation Controls")

score = st.number_input(
    "🏏 Score",
    min_value=0,
    max_value=300,
    value=114,
    step=1
)

over = st.number_input(
    "⏱️ Overs Completed",
    min_value=0.0,
    max_value=20.0,
    value=12.0,
    step=0.1
)

wickets = st.number_input(
    "❌ Wickets Lost",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)

run_rate = score / over if over > 0 else 0
required_rate = (target - score) / (20 - over) if over < 20 else 0

st.write(f"📊 Current Run Rate: {run_rate:.2f}")
st.write(f"📊 Required Run Rate: {required_rate:.2f}")

# --------------------------
# PREDICTION
# --------------------------
if st.button("🏏 Predict Match Result"):
    try:
        team_map = {
            "CSK": 1, "MI": 2, "RCB": 3, "KKR": 4, "SRH": 5,
            "RR": 6, "PBKS": 7, "GT": 8, "LSG": 9, "DC": 10
        }

        city_map = {
            "Mumbai": 1, "Chennai": 2, "Bangalore": 3, "Kolkata": 4,
            "Hyderabad": 5, "Delhi": 6, "Ahmedabad": 7, "Jaipur": 8
        }

        input_data = np.array([[
            team_map[batting_team],
            team_map[bowling_team],
            city_map[city],
            target,
            score,
            over,
            wickets
        ]])

        prediction = model.predict(input_data)

        st.subheader("🏏 Match Prediction Result")

        if prediction[0] > 0:
            st.success("🏆 Batting Team Advantage")
        else:
            st.error("🎯 Bowling Team Dominating")

        st.write(f"📊 Prediction Score: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("""
<hr style="margin-top:40px;">
<div style="
    text-align: center;
    color: gray;
    font-size: 14px;
    padding: 10px;
">
    🏏 Developed by <b>SHAIK MOHAMMED ALI</b>
</div>
""", unsafe_allow_html=True)
