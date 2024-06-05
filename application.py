from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import logging

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        message = request.form.get('message')
        if not message:
            return render_template('home.html', prediction="No message provided")
        
        try:
            data = CustomData(message=message)
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Received data for prediction: {pred_df}")

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            logging.info(f"Prediction result: {results}")
            return render_template('home.html', prediction=results[0])
        except ValueError as e:
            logging.error(f"Value error: {e}")
            return render_template('home.html', prediction=str(e))
        except Exception as e:
            logging.error(f"Exception: {e}")
            return render_template('home.html', prediction="An error occurred")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
