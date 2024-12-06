from flask import Flask, render_template, request
import os
import numpy as np

import pandas as pd

import joblib
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # Initializing a Flask app

# Load the label encoders globally to avoid loading them on every prediction request
encoder_path = 'artifacts/data_transformation/label_encoders.joblib' 
label_encoders = joblib.load(encoder_path)

@app.route('/', methods=['GET'])  # Route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # Route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user
            year = int(request.form['year'])
            km_driven = int(request.form['km_driven'])
            fuel = request.form['fuel']
            seller_type = request.form['seller_type']
            transmission = request.form['transmission']
            owner = request.form['owner']
            mileage = float(request.form['mileage'])
            engine = float(request.form['engine'])
            max_power = float(request.form['max_power'])
            seats = float(request.form['seats'])

            # Prepare the input data
            data = {
                "year": [year],
                "km_driven": [km_driven],
                "fuel": [fuel],
                "seller_type": [seller_type],
                "transmission": [transmission],
                "owner": [owner],
                "mileage(km/ltr/kg)": [mileage],
                "engine": [engine],
                "max_power": [max_power],
                "seats": [seats]
            }

            # Create DataFrame for the model
            input_df = pd.DataFrame(data)

            # Debugging step: Log the input DataFrame
            print("Input DataFrame:\n", input_df)

            # Encode categorical features using the saved label encoders
# Encode categorical features using the saved label encoders
            for column, encoder in label_encoders.items():
                if column in input_df.columns:
                    input_df[column] = input_df[column].astype(str).map(
                        lambda x: x if x in encoder.classes_ else 'unknown'  # Handle unknown values
                    )
                    encoder.classes_ = np.append(encoder.classes_, 'unknown')  # Add 'unknown' to encoder classes
                    input_df[column] = encoder.transform(input_df[column])

            # Debugging step: Log the encoded DataFrame
            print("Encoded DataFrame:\n", input_df)

            # Perform prediction using the prediction pipeline
            obj = PredictionPipeline()  # Ensure this class is defined and handles preprocessing
            predict = obj.predict(input_df)

            # Extract the numeric value and format it
            prediction_value = round(float(predict[0]), 2)

            return render_template('results.html', prediction=prediction_value)

        except Exception as e:
            # Debugging step: Log the exception
            print('The Exception message is:', e)
            return f"Something is wrong: {e}"

    else:
        return render_template('index.html')

# Running the app
if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=8080, debug=True)
    app.run(host="0.0.0.0", port=8080)
