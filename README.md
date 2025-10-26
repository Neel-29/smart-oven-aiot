# Smart Oven AIoT Project 🧑‍🍳🧠

This project is a proof-of-concept for an AIoT (Artificial Intelligence + Internet of Things) Smart Oven. It moves beyond traditional "dumb" ovens by using a machine learning model to predict optimal cooking times and temperatures based on rich contextual data.

The system features a closed-loop reinforcement learning pipeline, allowing it to learn from user feedback and improve its predictions over time.

## 1. The Problem

Traditional ovens are blunt instruments. They heat based on fixed presets, ignoring crucial variables like:
* The actual dish and its ingredients.
* The amount of food being cooked.
* The room's temperature and humidity.
* User-specific taste preferences.

This leads to overcooked cookies, undercooked pizza, and unpredictable results. This project transforms cooking from guesswork to data-driven, personalized perfection.

## 2. The Solution: AI + IoT

This project simulates a full AIoT ecosystem:
* **IoT (Sensors):** We simulate sensor data like `Room_Temp` and `Room_Humidity` as inputs for the model.
* **AI (The Brain):** A multi-input TensorFlow/Keras model predicts the `Oven_Temp` and `Oven_Duration`. Unlike a simple model, this one understands **ingredients** and **dish categories (tags)** to make intelligent, first-time predictions.
* **Reinforcement Learning (The Secret Sauce):** After every dish, the user provides feedback (✅ Perfect, 🤏 Undercooked, ❌ Overcooked). This feedback is logged and used by a **Reinforcement Engine** to fine-tune the model, making it smarter and more personalized with every use.

## 3. Architecture Overview 🏗️

This project consists of three main components that run simultaneously:

1.  **Backend API (Flask):** The "brain" that loads the trained AI model (`.h5`) and serves predictions via a REST API. It also has a `/feedback` endpoint to log user ratings to a database.
2.  **Frontend App (Streamlit):** An interactive web UI where the user can enter a dish name, get an AI-powered recommendation, and provide feedback on the results.
3.  **Reinforcement Engine (Python Script):** An offline script (`r1_engine.py`) that reads all the user feedback from the database, calculates "corrected" cooking parameters, and re-trains the model.

![A diagram showing the flow: User Input -> Streamlit Frontend -> Flask API -> AI Model -> Prediction -> User Feedback -> Database -> Reinforcement Engine -> New AI Model](https://i.imgur.com/g0P1S4c.png)

## 4. Technology Stack 🛠️

* **AI & Data Science:** TensorFlow (Keras), Pandas, NumPy, Scikit-learn, NLTK
* **Backend API:** Flask
* **Frontend UI:** Streamlit
* **Data Storage:** SQLite (for `oven_logs.db`)
* **Development:** Python, Jupyter Notebooks

## 5. How to Run This Project

Follow these steps to get the full system running locally.

### Step 1: Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/smart-oven-aiot.git
    cd smart-oven-aiot
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On Mac/Linux
    ```

3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Data Preparation & Model Training

You must run the Jupyter notebooks in order to create the dataset and train the initial AI model.

1.  **Run Notebook 1 (Data ETL):**
    * Launch Jupyter: `jupyter notebook`
    * Open `/notebooks/eda_and_extraction.ipynb`.
    * Run all cells from top to bottom. This will:
        * Parse the raw CSV files.
        * Extract oven temperatures and times using regex.
        * Merge ingredients, tags, and ratings.
        * Augment the data with simulated sensor values.
        * Save the final `processed_oven_recipes_v2.csv` file.

2.  **Run Notebook 2 (Model Training):**
    * Open `/notebooks/model_prototyping.ipynb`.
    * Run all cells from top to bottom. This will:
        * Load the `v2` dataset.
        * Preprocess all 4 inputs (Name, Env, Ingredients, Tags).
        * Build and train the multi-input Keras model.
        * Save the trained model (`oven_predictor_v2.h5`) and all 5 preprocessors (`.pkl`) to `/ml_model/models/`.

### Step 3: Run the Live System

You will need **two terminals** running at the same time.

1.  **Terminal 1: Run the Backend API:**
    ```bash
    cd api
    python app.py
    ```
    *(Leave this terminal running. It will load the model and serve predictions at `http://127.0.0.1:5000`)*

2.  **Terminal 2: Run the Frontend App:**
    ```bash
    # From the project root directory
    streamlit run frontend/app_frontend.py
    ```
    *(This will automatically open the Streamlit app in your browser.)*

### Step 4: Test the Reinforcement Loop

1.  **Use the App:** Go to the Streamlit app in your browser.
2.  **Get a Prediction:** Type in a dish name (e.g., "arriba baked winter squash mexican style") and click "Get AI Recommendation."
3.  **Give Feedback:** You'll see the AI's prediction. Click one of the feedback buttons (e.g., "Overcooked / Dry ❌"). This saves your feedback to `oven_logs.db`.
4.  **Stop your API** (Ctrl+C in Terminal 1).
5.  **Run the Engine:** In a terminal, run the reinforcement engine to re-train the model on your feedback:
    ```bash
    cd ml_model
    python r1_engine.py
    ```
    *(This will create a new, smarter `oven_predictor_v3.h5`)*
6.  **Update the API:**
    * Open `/api/app.py`.
    * Change the model path from `v2.h5` to `v3.h5`.
7.  **Relaunch the API** (Terminal 1) and test again. The prediction for that same dish will now be corrected based on your feedback. The loop is complete.