# In /6_frontend/app_frontend.py

import streamlit as st
import requests
import json

# --- 1. Configuration ---
st.set_page_config(page_title="Smart Oven AI Chef", layout="centered")
st.title("Smart Oven AI Chef 🧑‍🍳")
st.info("ℹ️ Note: All recommendations are for a 1-person serving size.")

# This is the URL of your local Flask API
API_URL = "http://127.0.0.1:5000"

# --- 2. Initialize Session State ---
# Session state is how Streamlit remembers variables between user interactions
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'dish_name' not in st.session_state:
    st.session_state.dish_name = "best chocolate chip cookies" # Pre-fill for easy testing
if 'env_data' not in st.session_state:
    st.session_state.env_data = {"room_temp": 22.0, "room_humidity": 55.0}


# --- 3. Main Cooking Interface ---
st.subheader("1. What are we cooking?")

# Get user inputs
dish_name = st.text_input("Dish Name:", st.session_state.dish_name)
col1, col2 = st.columns(2)
with col1:
    room_temp = st.slider("Simulated Room Temp (°C)", 15.0, 30.0, st.session_state.env_data['room_temp'])
with col2:
    room_humidity = st.slider("Simulated Room Humidity (%)", 30.0, 70.0, st.session_state.env_data['room_humidity'])

# "Cook!" button
if st.button("Get AI Recommendation", type="primary"):
    if not dish_name:
        st.error("Please enter a dish name.")
    else:
        with st.spinner("Asking the AI chef..."):
            try:
                # 1. Package the request data
                payload = {
                    "dish_name": dish_name,
                    "room_temp": room_temp,
                    "room_humidity": room_humidity
                }
                
                # 2. Call the /predict endpoint
                response = requests.post(f"{API_URL}/predict", json=payload)
                
                if response.status_code == 200:
                    # 3. Store the prediction in the session state
                    st.session_state.prediction = response.json()
                    st.session_state.dish_name = dish_name # Remember the dish
                    st.session_state.env_data = {"room_temp": room_temp, "room_humidity": room_humidity}
                else:
                    st.error(f"Error from API: {response.json().get('error', 'Unknown error')}")
            
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Is the Flask API running in the other terminal?")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- 4. Display Prediction & Get Feedback ---
if st.session_state.prediction:
    pred = st.session_state.prediction
    temp = pred['predicted_temp']
    duration = pred['predicted_duration']
    
    st.subheader("2. AI Recommendation")
    st.info(f"The AI recommends cooking **{st.session_state.dish_name}** at **{temp:.0f}°F** for **{duration:.0f} minutes**.")
    
    # This is the problem you found!
    if duration > 60 or temp > 450:
         st.warning("⚠️ That seems... wrong. This is the 'blunt instrument' problem!")

    st.subheader("3. How did it turn out? (Your Feedback)")
    st.write("This is the 'Secret Sauce' (Step 5). Your rating will teach the AI.")
    
    # --- Feedback Button Logic ---
    col1, col2, col3 = st.columns(3)
    
    def send_feedback(feedback_value):
        try:
            # Package the prediction data *and* the feedback
            payload = {
                "dish_name": st.session_state.dish_name,
                "room_temp": st.session_state.env_data['room_temp'],
                "room_humidity": st.session_state.env_data['room_humidity'],
                "predicted_temp": st.session_state.prediction['predicted_temp'],
                "predicted_duration": st.session_state.prediction['predicted_duration'],
                "user_feedback": feedback_value  # -1, 0, or 1
            }
            
            # Call the /feedback endpoint
            response = requests.post(f"{API_URL}/feedback", json=payload)
            
            if response.status_code == 200:
                st.success("✅ Feedback saved! The AI will learn from this.")
            else:
                st.error("Failed to save feedback.")
                
            # Clear the prediction to reset the app
            st.session_state.prediction = None
            
        except Exception as e:
            st.error(f"Error sending feedback: {e}")

    with col1:
        if st.button("Overcooked / Dry ❌"):
            send_feedback(-1)
            
    with col2:
        if st.button("Slightly undercooked 🤏"):
            send_feedback(0)
            
    with col3:
        if st.button("Perfect ✅"):
            send_feedback(1)