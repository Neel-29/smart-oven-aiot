# In /api/app.py (This is the new V2 file)

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
import ast
import io
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import mobilenet_v2

# --- 1. Initialize Flask App & Database (ROBUST PATHS) ---
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root (e.g., .../smart-oven-aiot)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

# Define paths based on the project root
# This will correctly resolve to e.g., "D:\smart-oven-aiot\data\oven_logs.db"
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'oven_logs.db')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'ml_model', 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

print(f"Project Root: {PROJECT_ROOT}")
print(f"Model Dir: {MODEL_DIR}")
print(f"Data Dir: {DATA_DIR}")
print(f"DB Path: {DB_PATH}")
# --- 2. Load Data for "Context Lookup" ---
# This is our "recipe book" that the API uses to find ingredients/tags
try:
    print("Loading recipe lookup data...")
    # Load processed data (has ingredients and tags)
    lookup_path = os.path.join(DATA_DIR, 'processed', 'processed_oven_recipes_v2.csv')
    recipe_lookup = pd.read_csv(lookup_path)
    
    # Convert string-lists back to real lists for the binarizers
    recipe_lookup['ingredient_ids'] = recipe_lookup['ingredient_ids'].apply(ast.literal_eval)
    recipe_lookup['tags'] = recipe_lookup['tags'].apply(ast.literal_eval)
    
    # Set 'name' as the index for fast lookups
    recipe_lookup.set_index('name', inplace=True)
    print(f"Loaded {len(recipe_lookup)} recipes into lookup table.")

except Exception as e:
    print(f"CRITICAL ERROR: Could not load recipe lookup data. {e}")
    recipe_lookup = None


# --- 3. Load V2 Models & Preprocessors ---
# We now load the V2 model and all 5 preprocessors
try:
    print("Loading AI v2 model and preprocessors...")
    model_path = os.path.join(MODEL_DIR, 'oven_predictor_v2.h5')
    model = tf.keras.models.load_model(model_path, compile=False) # compile=False is still needed
    
    with open(os.path.join(MODEL_DIR, 'name_encoder_v2.pkl'), 'rb') as f:
        name_encoder = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'env_scaler_v2.pkl'), 'rb') as f:
        env_scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'output_scaler_v2.pkl'), 'rb') as f:
        output_scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'ingredient_binarizer.pkl'), 'rb') as f:
        ingredient_binarizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'tag_binarizer.pkl'), 'rb') as f:
        tag_binarizer = pickle.load(f)
        
    print("All V2 models loaded successfully.")

    print("Loading CV model...")
    cv_model_path = os.path.join(MODEL_DIR, 'dish_classifier_v1.h5')
    cv_model = tf.keras.models.load_model(cv_model_path, compile=False)
    
    # Load the class names
    class_names_path = os.path.join(MODEL_DIR, 'food_101_class_names.txt')
    with open(class_names_path, 'r') as f:
        food_101_class_names = [line.strip() for line in f.readlines()]
    print("CV model loaded.")

except Exception as e:
    print(f"CRITICAL ERROR: Could not load models. {e}")
    model = None
    cv_model = None


# Helper function (no changes)
def init_db():
    print(f"Initializing database at {DB_PATH}")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        dish_name TEXT NOT NULL,
        room_temp REAL,
        room_humidity REAL,
        predicted_temp REAL,
        predicted_duration REAL,
        user_feedback INTEGER
    )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")


# --- 4. NEW Prediction Helper Function (V2) ---
def make_prediction_v2(dish_name, room_temp, room_humidity):
    """
    Uses the loaded V2 models to make a single smart prediction.
    """
    if model is None:
        raise Exception("Model is not loaded.")
    if recipe_lookup is None:
        raise Exception("Recipe lookup data is not loaded.")
        
    # --- 1. Clean the user's input ---
    query_name = dish_name.strip().lower() # Remove spaces and go to lowercase
    if not query_name:
        raise Exception("Dish name is empty.")

    # --- 2. Look up context (ingredients and tags) ---
    
    # Try a simple "contains" search first. This is the new fuzzy logic.
    # This will find "arriba..." from the query "baked winter..."
    possible_matches = recipe_lookup[recipe_lookup.index.str.lower().str.contains(query_name)]
    
    if possible_matches.empty:
        # If no "contains" match, try an exact match (this is less likely)
        try:
            recipe_data = recipe_lookup.loc[query_name]
        except KeyError:
            # No match found at all
            raise Exception(f"Dish '{dish_name}' not found in recipe database.")
    else:
        # We found one or more "contains" matches. Just use the first one.
        recipe_data = possible_matches.iloc[0]

    # At this point, recipe_data is a single row (a pandas Series)
    # from our recipe_lookup.
    
    print(f"User query '{dish_name}' matched to recipe: {recipe_data.name}")

    # --- 3. Process ALL 4 Inputs ---
    # Input A: Name (Use the *actual name* from the DB)
    sample_name_encoded = name_encoder.transform([[recipe_data.name]]) 
    # Input B: Environment
    sample_env = [[room_temp, room_humidity]]
    sample_env_scaled = env_scaler.transform(sample_env)
    # Input C: Ingredients
    sample_ingredients = [recipe_data['ingredient_ids']]
    sample_ingr_encoded = ingredient_binarizer.transform(sample_ingredients)
    # Input D: Tags
    sample_tags = [recipe_data['tags']]
    sample_tags_encoded = tag_binarizer.transform(sample_tags)

    # --- 4. Package for prediction ---
    X_pred_list = [sample_name_encoded, sample_env_scaled, sample_ingr_encoded, sample_tags_encoded]

    # --- 5. Make Prediction ---
    scaled_pred_temp, scaled_pred_duration = model.predict(X_pred_list)

    # --- 6. Invert Scaling (to get real-world values) ---
    scaled_pred = np.array([[scaled_pred_temp[0][0], scaled_pred_duration[0][0]]])
    final_prediction = output_scaler.inverse_transform(scaled_pred)
    
    return final_prediction[0][0], final_prediction[0][1]

def load_and_prep_image(img_bytes):
    # Load image from bytes
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    # Convert to array
    img_array = image.img_to_array(img)
    # Expand dimensions to create a batch of 1
    img_array_expanded = np.expand_dims(img_array, axis=0)
    # Preprocess the image for MobileNetV2
    return mobilenet_v2.preprocess_input(img_array_expanded)

# --- 5. Define API Endpoints ---

@app.route('/')
def home():
    return "Smart Oven AIoT API (V2 - Smart) is running."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to get a new cooking prediction (V2).
    """
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500
        
    try:
        data = request.get_json()
        dish_name = data['dish_name']
        room_temp = data.get('room_temp', 20.0)
        room_humidity = data.get('room_humidity', 50.0)

        # Get predictions from our NEW V2 function
        pred_temp, pred_duration = make_prediction_v2(dish_name, room_temp, room_humidity)
        
        return jsonify({
            'predicted_temp': int(round(pred_temp, 0)),
            'predicted_duration': int(round(pred_duration, 0))
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# The /feedback endpoint doesn't need any changes!
# It just logs the data, which is perfect.
@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Endpoint to log user feedback for reinforcement learning.
    """
    try:
        data = request.get_json()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback_log (
                dish_name, room_temp, room_humidity, 
                predicted_temp, predicted_duration, user_feedback
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (data['dish_name'], data.get('room_temp', 20.0), 
              data.get('room_humidity', 50.0), data['predicted_temp'], 
              data['predicted_duration'], data['user_feedback']))
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success", "message": "Feedback logged."})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/classify_image', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and cv_model:
        try:
            # 1. Read and prep the image
            img_bytes = file.read()
            prepped_image = load_and_prep_image(img_bytes)
            
            # 2. Make CV prediction
            predictions = cv_model.predict(prepped_image)
            predicted_index = np.argmax(predictions[0])
            dish_name = food_101_class_names[predicted_index]
            
            # Replace underscores with spaces (e.g., "pork_chop" -> "pork chop")
            dish_name = dish_name.replace("_", " ")
            
            print(f"CV Model classified image as: {dish_name}")
            
            # 3. Call your *existing* prediction function
            # We use default room temp/humidity for this example
            pred_temp, pred_duration = make_prediction_v2(dish_name, 20.0, 50.0)

            # 4. Return the final, combined prediction
            return jsonify({
                'classified_dish': dish_name,
                'predicted_temp': int(round(pred_temp, 0)),
                'predicted_duration': int(round(pred_duration, 0))
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    return jsonify({"error": "CV model not loaded"}), 500

# --- 6. Run the Application ---
if __name__ == '__main__':
    init_db()  # Create the database and table on first run
    app.run(debug=True, port=5000)