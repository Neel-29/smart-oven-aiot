import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
import ast

print("--- Smart Reinforcement Engine (V2) ---")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'oven_logs.db')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'ml_model', 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, 'oven_predictor_v2.h5')
NEW_MODEL_PATH = os.path.join(MODEL_DIR, 'oven_predictor_v3.h5')

try:
    conn = sqlite3.connect(DB_PATH)
    df_feedback = pd.read_sql_query("SELECT * FROM feedback_log", conn)
    conn.close()
    
    if df_feedback.empty:
        print("No new feedback found in database. Exiting.")
        exit()
        
    print(f"Loaded {len(df_feedback)} new feedback entries.")
except Exception as e:
    print(f"Error loading feedback: {e}")
    exit()

try:
    print("Loading recipe lookup data...")
    lookup_path = os.path.join(DATA_DIR, 'processed', 'processed_oven_recipes_v2.csv')
    recipe_lookup = pd.read_csv(lookup_path)
    recipe_lookup['ingredient_ids'] = recipe_lookup['ingredient_ids'].apply(ast.literal_eval)
    recipe_lookup['tags'] = recipe_lookup['tags'].apply(ast.literal_eval)
    recipe_lookup.set_index('name', inplace=True)
    print("Recipe lookup loaded.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load recipe lookup data. {e}")
    exit()

def generate_corrected_targets(row):
    temp = row['predicted_temp']
    duration = row['predicted_duration']
    feedback = row['user_feedback']
    
    if feedback == 1: 
        return temp, duration
    elif feedback == 0:
        return temp, duration * 1.15 
    elif feedback == -1: 
        return temp * 0.98, duration * 0.85
    
df_feedback[['Corrected_Temp', 'Corrected_Duration']] = df_feedback.apply(
    generate_corrected_targets, axis=1, result_type='expand'
)
print("Applied correction logic to feedback.")
print(df_feedback[['dish_name', 'predicted_duration', 'Corrected_Duration']].head())

print("Loading V2 preprocessors and model...")
try:
    model = tf.keras.models.load_model(CURRENT_MODEL_PATH)
    
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
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

print("Creating new training batch...")
X_name_list, X_env_list, X_ingr_list, X_tags_list = [], [], [], []

for index, row in df_feedback.iterrows():
    dish_name = row['dish_name']
    
    try:
        recipe_data = recipe_lookup.loc[dish_name].iloc[0]
    except Exception:
        possible_matches = recipe_lookup[recipe_lookup.index.str.lower().str.contains(dish_name.lower())]
        if possible_matches.empty:
            print(f"Warning: Could not find ingredients for '{dish_name}'. Skipping.")
            continue
        recipe_data = possible_matches.iloc[0]

    X_name_list.append(name_encoder.transform([[recipe_data.name]]))
    X_env_list.append(env_scaler.transform([[row['room_temp'], row['room_humidity']]]))
    X_ingr_list.append(ingredient_binarizer.transform([recipe_data['ingredient_ids']]))
    X_tags_list.append(tag_binarizer.transform([recipe_data['tags']]))

X_name_new = np.concatenate(X_name_list, axis=0)
X_env_new = np.concatenate(X_env_list, axis=0)
X_ingr_new = np.concatenate(X_ingr_list, axis=0)
X_tags_new = np.concatenate(X_tags_list, axis=0)

X_train_list = [X_name_new, X_env_new, X_ingr_new, X_tags_new]

Y_new = df_feedback[['Corrected_Temp', 'Corrected_Duration']]
Y_new_scaled = output_scaler.transform(Y_new)
Y_temp_new = Y_new_scaled[:, 0]
Y_duration_new = Y_new_scaled[:, 1]
Y_train_list = [Y_temp_new, Y_duration_new]

print(f"Fine-tuning model on {len(X_name_new)} new data points...")
history = model.fit(
    X_train_list,
    Y_train_list,
    epochs=15,  
    batch_size=1,
    verbose=1
)
print("Fine-tuning complete.")

model.save(NEW_MODEL_PATH)
print(f"✨ New, smarter model saved as: {NEW_MODEL_PATH}")

conn = sqlite3.connect(DB_PATH)
conn.execute("DELETE FROM feedback_log")
conn.commit()
conn.close()
print("Feedback log cleared. Ready for new feedback.")