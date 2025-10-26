# Smart Oven AIoT Project 🧑‍🍳🧠📸

This project is a proof-of-concept for an AIoT (Artificial Intelligence + Internet of Things) Smart Oven. It moves beyond traditional "dumb" ovens by using machine learning models to predict optimal cooking times/temperatures based on rich contextual data and **automatically recognize dishes using a camera**.

The system features a closed-loop reinforcement learning pipeline, allowing it to learn from user feedback and improve its predictions over time.

## 1. The Problem

Traditional ovens are blunt instruments. They heat based on fixed presets, ignoring crucial variables like:
* The actual dish and its ingredients.
* The amount of food being cooked.
* The room's temperature and humidity.
* User-specific taste preferences.

This leads to overcooked cookies, undercooked pizza, and unpredictable results. This project transforms cooking from guesswork to data-driven, personalized perfection.

## 2. The Solution: AI + IoT + Vision 👁️

This project simulates a full AIoT ecosystem:
* **IoT (Sensors):** We simulate sensor data like `Room_Temp` and `Room_Humidity` as inputs.
* **Computer Vision:** A camera (simulated via file upload) automatically recognizes the dish using a fine-tuned MobileNetV2 model trained on Food-101.
* **AI (Prediction Brain):** A multi-input TensorFlow/Keras model predicts the `Oven_Temp` and `Oven_Duration`. It uses the **recognized dish name**, **ingredients**, **tags**, and sensor data for intelligent, first-time predictions.
* **Reinforcement Learning (The Secret Sauce):** After every dish, the user provides feedback (✅ Perfect, 🤏 Undercooked, ❌ Overcooked). This feedback is logged and used by a **Reinforcement Engine** to fine-tune the prediction model, making it smarter and more personalized.

## 3. Architecture Overview 🏗️

The system now includes a separate Computer Vision model that feeds into the main prediction pipeline.

1.  **Input:** User provides an image OR types a dish name.
2.  **Frontend (Streamlit):** Sends the input to the appropriate API endpoint.
3.  **Backend API (Flask):**
    * `/classify_image`: Receives an image, uses the **CV Model** to get a `dish_name`.
    * `/predict`: Receives a `dish_name` (either from CV or manual input) and sensor values. It looks up the dish's ingredients/tags.
    * Both endpoints then feed the `dish_name`, `ingredients`, `tags`, and sensor values into the **Prediction Model** to get `Temp` and `Duration`.
    * `/feedback`: Logs user ratings to a database.
4.  **Reinforcement Engine (Python Script):** Offline script reads feedback, calculates corrections, and re-trains the **Prediction Model**.

## 4. Technology Stack 🛠️

* **AI & Data Science:** TensorFlow (Keras), Pandas, NumPy, Scikit-learn, NLTK, TensorFlow Datasets, OpenCV (headless)
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

You must run the Jupyter notebooks in order to create the datasets and train the AI models.

1.  **Run Notebook 1 (Prediction Data ETL):**
    * Launch Jupyter: `jupyter notebook`
    * Open `/notebooks/eda_and_extraction.ipynb`.
    * Run all cells from top to bottom. This creates `processed_oven_recipes_v2.csv`.

2.  **Run Notebook 2 (Prediction Model Training):**
    * Open `/notebooks/model_prototyping.ipynb`.
    * Run all cells from top to bottom. This trains the prediction model and saves `oven_predictor_vX.h5` and its preprocessors.

3.  **Prepare Food-101 Data (CV Model):**
    * **Manual Download:** Download `food-101.tar.gz` from [http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz).
    * **Extract:** Use WinRAR/7-Zip to extract the archive.
    * **Move:** Find the `images` folder inside the extracted `food-101` folder. Move this `images` folder to `D:\smart-oven-aiot\data\raw\`. The final path should be `D:\smart-oven-aiot\data\raw\images\`.

4.  **Run Notebook 3 (CV Model Training):**
    * Open `/notebooks/03_computer_vision_model.ipynb`.
    * **Important:** Ensure the code uses the **manual loading** method (`tf.keras.utils.image_dataset_from_directory`) pointing to `../data/raw/images`.
    * Run all cells. This will train the dish classifier and save `dish_classifier_v1.h5` and `food_101_class_names.txt`.

### Step 3: Run the Live System

You will need **two terminals** running at the same time.

1.  **Terminal 1: Run the Backend API:**
    ```bash
    cd api
    python app.py
    ```
    *(Ensure `app.py` is configured to load the latest prediction model, e.g., `v3.h5`. It will also load the CV model.)*

2.  **Terminal 2: Run the Frontend App:**
    ```bash
    # From the project root directory
    streamlit run frontend/app_frontend.py
    ```
    *(This will open the Streamlit app in your browser.)*

### Step 4: Use the App & Reinforcement Loop

1.  **Choose Input:** Select "Upload Image" or "Enter Name Manually".
2.  **Provide Input:** Upload an image or type a name.
3.  **Adjust Sensors:** Modify the simulated temperature/humidity sliders.
4.  **Get Final Recommendation:** Click the button to see the AI's prediction based on all inputs.
5.  **Give Feedback:** Click one of the feedback buttons (✅ 🤏 ❌).
6.  **(Offline) Retrain:** Stop the API, run `python ml_model/r1_engine.py` to create the next model version (e.g., `v4.h5`), update `api/app.py` to use `v4.h5`, and restart the API. The system is now smarter.