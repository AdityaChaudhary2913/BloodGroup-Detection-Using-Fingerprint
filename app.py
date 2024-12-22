from flask import Flask, render_template, url_for, request, session, jsonify, redirect
from bloodgroup.exception import CustomException
from bloodgroup.constants import *
import sys, os, cv2
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

def preprocess_image(filepath):
    try:
        # Read the image
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

        # Crop the central region to isolate the fingerprint
        height, width = image.shape
        crop_size = min(height, width)  # Square crop
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        cropped = image[start_y:start_y + crop_size, start_x:start_x + crop_size]

        # Resize the image to match dataset dimensions (e.g., 96x96)
        desired_width, desired_height = 96, 96  # Replace with your dataset's dimensions
        resized = cv2.resize(cropped, (desired_width, desired_height), interpolation=cv2.INTER_AREA)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)

        # Enhance fingerprint details using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Convert to binary image (thresholding)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological operations to refine the fingerprint
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Save the preprocessed image
        preprocessed_path = filepath.replace('captured_image.png', 'preprocessed_image.png')
        cv2.imwrite(preprocessed_path, morphed)

        return preprocessed_path
    except Exception as e:
        raise CustomException(f"Error in preprocessing image: {str(e)}", sys)
    
@app.route("/preprocess", methods=['POST'])
def preprocess_image_api():
    try:
        if 'file' not in request.files:
            return jsonify(error="No file uploaded"), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify(error="No file selected"), 400
        
        # Save the uploaded file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"preprocessed_{file.filename}")
        file.save(filepath)

        preprocessed_path = preprocess_image(filepath)

        # Return the preprocessed image path
        return jsonify(path=f"/{preprocessed_path}")
    except Exception as e:
        return jsonify(error=str(e)), 500

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
            # os.remove(filepath)
            return jsonify(result=prediction, file=file.filename)
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise CustomException(e, sys)
    return render_template('image_classifier.html')

    
if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug= True)