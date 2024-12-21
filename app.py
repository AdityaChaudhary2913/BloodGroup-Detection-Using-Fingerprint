from flask import Flask, render_template, url_for, request, session, jsonify, redirect
from bloodgroup.exception import CustomException
from bloodgroup.constants import *
import sys, os
from bloodgroup.pipeline.training_pipeline import TrainPipeline
from bloodgroup.pipeline.prediction_pipeline import PredictionPipeline


app = Flask(__name__)
app.secret_key = os.getenv("SessionSecretKey")

ADMIN_ID = os.getenv("AdminID")
ADMIN_PASSWORD = os.getenv("AdminPassword")

# Directory to temporarily save uploaded images
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/train")
def train_route():
    if not session.get('admin_logged_in'):
        return redirect(url_for('home')) 
    try:
        if 'is_training' in session and session['is_training']:
            print("Training is already in progress.")
            return "Training is already in progress."
        else:
            session['is_training'] = True
            train_pipeline = TrainPipeline()
            train_pipeline.run_pipeline()
            session['is_training'] = False
        return render_template('training.html')
    except Exception as e:
        session['is_training'] = False
        raise CustomException(e,sys)


@app.route("/admin_login", methods=['POST'])
def admin_login():
    data = request.get_json()
    admin_id = data.get('adminID')
    admin_password = data.get('adminPassword')
    if admin_id == ADMIN_ID and admin_password == ADMIN_PASSWORD:
        session['admin_logged_in'] = True
        return jsonify(success=True)
    else:
        session['admin_logged_in'] = False
        return jsonify(success=False)

@app.route("/logout")
def logout():
    session['admin_logged_in'] = False
    session.pop('admin_logged_in', None)
    session.pop('is_training', None)
    return redirect(url_for('home'))

@app.route("/image_classifier", methods=['POST', 'GET'])
def image_classifier():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify(error="No file uploaded"), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify(error="No file selected"), 400
        
        # Save the file to the upload directory
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            prediction_pipeline = PredictionPipeline()
            prediction = prediction_pipeline.run_pipeline(filepath)
            os.remove(filepath)
            return jsonify(result=prediction, file=file.filename)
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise CustomException(e, sys)
    return render_template('image_classifier.html')

@app.route("/capture", methods=['POST'])
def capture_image():
    try:
        # Access the image sent from the camera
        image_data = request.files['file']

        # Save the captured image temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.png')
        image_data.save(filepath)

        # Use the prediction pipeline to classify the image
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.run_pipeline(filepath)
        
        # Delete the temporary file
        os.remove(filepath)

        return jsonify(result=prediction)
    except Exception as e:
        raise CustomException(e, sys)

    
if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug= True)