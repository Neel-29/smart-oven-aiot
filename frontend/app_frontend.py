import streamlit as st
import requests
import json
import io
import time
import streamlit.components.v1 as components

# --- 1. Configuration ---
st.set_page_config(page_title="Smart Oven AI Chef", layout="centered")
st.title("Smart Oven AI Chef 🧑‍🍳")
API_URL = "http://127.0.0.1:5000"

# --- 2. Initialize Session State ---
# General state
if 'input_method' not in st.session_state:
    st.session_state.input_method = "Enter Name Manually" # Default
if 'dish_name' not in st.session_state:
    st.session_state.dish_name = None
if 'initial_prediction' not in st.session_state:
    st.session_state.initial_prediction = None
if 'final_prediction' not in st.session_state:
    st.session_state.final_prediction = None
if 'sensor_temp' not in st.session_state:
    st.session_state.sensor_temp = 22.0
if 'sensor_humidity' not in st.session_state:
    st.session_state.sensor_humidity = 55.0

# Image specific state
if 'uploaded_image_bytes' not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if 'image_display_caption' not in st.session_state:
    st.session_state.image_display_caption = None

# --- Helper Function to Reset State ---
def reset_state():
    st.session_state.dish_name = None
    st.session_state.initial_prediction = None
    st.session_state.final_prediction = None
    st.session_state.uploaded_image_bytes = None
    st.session_state.image_display_caption = None
    # Keep sensor values unless explicitly reset elsewhere

# --- 3. Ask for Input Method ---
st.subheader("1. Choose Input Method")
input_method = st.radio(
    "How do you want to specify the dish?",
    ("Enter Name Manually", "Upload Image"),
    key='input_method_radio',
    on_change=reset_state # Reset if method changes
)
st.session_state.input_method = input_method # Update state

# --- 4. Input Section (Conditional) ---
st.subheader("2. Provide Dish Information")

if st.session_state.input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image of your dish...",
        type=["jpg", "jpeg", "png"],
        key="file_uploader_widget",
        on_change=reset_state # Reset if a new file is uploaded
    )

    if uploaded_file is not None and st.session_state.dish_name is None: # Only process if dish_name isn't set yet
        st.session_state.uploaded_image_bytes = uploaded_file.getvalue() # Store image bytes
        with st.spinner("Classifying image..."):
            try:
                files = {'file': (uploaded_file.name, st.session_state.uploaded_image_bytes, uploaded_file.type)}
                response = requests.post(f"{API_URL}/classify_image", files=files)

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.dish_name = data['classified_dish'] # Store dish name
                    st.session_state.initial_prediction = data # Store initial prediction
                    st.session_state.image_display_caption = f"Classified as: {st.session_state.dish_name}"
                    st.success(f"AI identified dish as: **{st.session_state.dish_name}**")
                else:
                    st.error(f"Error from API: {response.json().get('error', 'Unknown error')}")
                    reset_state()
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Is the Flask API running?")
                reset_state()
            except Exception as e:
                st.error(f"An error occurred during classification: {e}")
                reset_state()

elif st.session_state.input_method == "Enter Name Manually":
    manual_dish_name = st.text_input(
        "Enter Dish Name:",
        key="manual_dish_input",
        # Clear predictions if text changes significantly? Maybe not needed.
    )
    if st.button("Confirm Dish Name", key="confirm_manual_dish"):
        if manual_dish_name:
            st.session_state.dish_name = manual_dish_name
            # Need to get an initial prediction using default sensors
            with st.spinner("Getting initial recommendation..."):
                try:
                    payload = {"dish_name": st.session_state.dish_name} # Use default sensors
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    if response.status_code == 200:
                        st.session_state.initial_prediction = response.json()
                    else:
                        st.error(f"Error from API: {response.json().get('error', 'Could not get initial prediction')}")
                        reset_state()
                except Exception as e:
                    st.error(f"Error getting initial prediction: {e}")
                    reset_state()
        else:
            st.warning("Please enter a dish name.")

# --- 5. Display Image (if applicable) and Sensor Adjustment ---
if st.session_state.dish_name:
    st.subheader("3. Adjust Sensor Values")

    # Display image if uploaded
    if st.session_state.uploaded_image_bytes:
        st.image(st.session_state.uploaded_image_bytes, caption=st.session_state.image_display_caption, use_column_width=True)

    st.write(f"Selected dish: **{st.session_state.dish_name}**")

    # Display initial prediction if available
    if st.session_state.initial_prediction:
         st.info(f"Initial AI Suggestion (Default Sensors): **{st.session_state.initial_prediction['predicted_temp']}°F** for **{st.session_state.initial_prediction['predicted_duration']} minutes**.")

    # Sensor sliders
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.sensor_temp = st.slider(
            "Simulated Room Temp (°C)", 15.0, 30.0, st.session_state.sensor_temp, key="slider_temp"
        )
    with col2:
        st.session_state.sensor_humidity = st.slider(
            "Simulated Room Humidity (%)", 30.0, 70.0, st.session_state.sensor_humidity, key="slider_humidity"
        )

    # --- 6. Get Final Recommendation ---
    if st.button("Get Final Recommendation", key="get_final_rec", type="primary"):
        with st.spinner("Calculating final recommendation based on sensors..."):
            try:
                # Call the standard /predict endpoint with the dish name and CURRENT sensor values
                payload = {
                    "dish_name": st.session_state.dish_name,
                    "room_temp": st.session_state.sensor_temp,
                    "room_humidity": st.session_state.sensor_humidity
                }
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    st.session_state.final_prediction = response.json()
                else:
                    st.error(f"Error from API: {response.json().get('error', 'Could not get final prediction')}")
                    st.session_state.final_prediction = None
            except Exception as e:
                st.error(f"Error getting final prediction: {e}")
                st.session_state.final_prediction = None

# --- 7. Display Final Result & Feedback ---
if st.session_state.final_prediction:
    st.subheader("4. Final AI Recommendation")
    final_pred = st.session_state.final_prediction
    st.success(f"Cook **{st.session_state.dish_name}** at **{final_pred['predicted_temp']}°F** for **{final_pred['predicted_duration']} minutes** (considering room temp {st.session_state.sensor_temp}°C and humidity {st.session_state.sensor_humidity}%).")

    # --- Feedback section ---
    st.subheader("5. How did it turn out?")
    fb_col1, fb_col2, fb_col3 = st.columns(3)

    def send_final_feedback(feedback_value):
        try:
            payload = {
                "dish_name": st.session_state.dish_name,
                "room_temp": st.session_state.sensor_temp,
                "room_humidity": st.session_state.sensor_humidity,
                "predicted_temp": st.session_state.final_prediction['predicted_temp'],
                "predicted_duration": st.session_state.final_prediction['predicted_duration'],
                "user_feedback": feedback_value
            }
            response = requests.post(f"{API_URL}/feedback", json=payload)

            if response.status_code == 200:
                st.success("✅ Feedback saved! The AI will learn.")
                st.session_state.feedback_submitted = True
            else:
                st.error("Failed to save feedback.")

        except Exception as e:
            st.error(f"Error sending feedback: {e}")

    with fb_col1:
        if st.button("Overcooked / Dry ❌", key="final_overcooked"): send_final_feedback(-1)
    with fb_col2:
        if st.button("Slightly undercooked 🤏", key="final_undercooked"): send_final_feedback(0)
    with fb_col3:
        if st.button("Perfect ✅", key="final_perfect"): send_final_feedback(1)

# --- Add a reset button for convenience ---
st.divider()
if st.button("Start Over", key="reset_all"):
    reset_state()
    st.session_state.feedback_submitted = False
    components.html(
        """
        <script>
            window.parent.location.reload();
        </script>
        """,
        height=0
    )