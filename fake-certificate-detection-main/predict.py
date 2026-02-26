import joblib
import pandas as pd
from feature_extractor import extract_features_single  # check if this function exists

# Load trained model
model = joblib.load("certificate_detection_model.pkl")

def predict_certificate(image_path):
    # Extract features from image
    features = extract_features_single(image_path)

    # Convert to DataFrame
    df = pd.DataFrame([features])

    # Predict
    prediction = model.predict(df)[0]

    if prediction == 1:
        return "Genuine"
    else:
        return "Fake"