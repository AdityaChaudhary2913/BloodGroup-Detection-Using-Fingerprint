**Project Overview**

 - **Name:** Blood Group Detection Using Fingerprint
 - **Purpose:** A Flask-based web application and training pipeline that predicts a person's blood group from a fingerprint image using a PyTorch model.
 - **Key features:** Image preprocessing, training pipeline, prediction pipeline, and a simple web UI for uploading images and starting model training (admin only).

**Quick Links**

 - **Demo Video:** https://drive.google.com/file/d/1fYMMivoZkwdRhf4-WesmlYpjE0dLUJJX/view?usp=sharing

**Repository Structure (important files)**

 - `app.py`: Flask application and routes for home, training and image classification.
 - `requirements.txt`: Python dependencies for the project.
 - `setup.py`: Package metadata used by pip editable install (`-e .`).
 - `bloodgroup/`: Core package with pipelines and components for data ingestion, transformation, model creation, training and prediction.
 - `artifacts/`: Trained models, data loaders and pipeline artifacts generated during experiments and runs.

**Requirements**

 - **Python:** 3.8+ (recommended), with `pip` available.
 - Install dependencies:

 ```bash
 pip install -r requirements.txt
 ```

**Environment variables**

The Flask app reads a few environment variables used for configuration and admin access. Set them before running the app:

 - `APP_HOST` : Host for Flask app (default in code if not provided).
 - `APP_PORT` : Port for Flask app.
 - `SessionSecretKey` : Flask `secret_key` for sessions.
 - `AdminID` and `AdminPassword` : Credentials required to access the `/train` admin route.

Example (zsh):

```bash
export APP_HOST=0.0.0.0
export APP_PORT=5000
export SessionSecretKey="your_secret_here"
export AdminID="admin"
export AdminPassword="password"
python app.py
```

**Running the app**

 - Start the Flask server:

 ```bash
 python app.py
 ```

 - Open the web UI in your browser at `http://<APP_HOST>:<APP_PORT>/` (for local runs default is `http://127.0.0.1:5000/`).

 - Use the **Image Classifier** page to upload fingerprint images (`templates/image_classifier.html`). The app will run the `PredictionPipeline` defined in `bloodgroup/pipeline/prediction_pipeline.py` and return the predicted blood group.

 - The `/train` route will start training via the `TrainPipeline` in `bloodgroup/pipeline/training_pipeline.py`. This route is protected by admin session — you must POST valid admin credentials to `/admin_login` to mark the session as logged in.

**Model & Artifacts**

 - Final trained model files live under `artifacts/` and `FinalModel/` (example: `artifacts/FinalModel/final_model_after_training.pth`).
 - The model referenced by the web app is expected at the path defined in `bloodgroup/constants` (check `FINAL_MODEL_PATH` and `FINAL_MODEL_AFTER_EVALUATION_NAME`).
 - A downloadable model (provided link): https://drive.google.com/file/d/1fYMMivoZkwdRhf4-WesmlYpjE0dLUJJX/view?usp=sharing

**Notes on Preprocessing**

 - `app.py` contains a `preprocess_image` function used by an API route `/preprocess` which converts uploaded images into a preprocessed binary fingerprint image and stores it in the configured upload folder.

**Development & Packaging**

 - The project contains a `setup.py` and is installable in editable mode. Install locally with:

 ```bash
 pip install -e .
 ```

**Troubleshooting**

 - If the model cannot be loaded, verify the model path in `bloodgroup/constants.py` and ensure the model file exists (or download from the Drive link above).
 - If image uploads fail, confirm `UPLOAD_FOLDER` exists and the Flask process has write permissions.
