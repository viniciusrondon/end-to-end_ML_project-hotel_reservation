import joblib
import numpy as np
from config.paths_config import *
from flask import Flask, request, render_template

app = Flask(__name__)

model = joblib.load(MODEL_OUTPUT_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        lead_time = np.int64(request.form["lead_time"])
        no_of_special_requests = np.int64(request.form["no_of_special_requests"])
        avg_price_per_room = np.float64(request.form["avg_price_per_room"])
        arrival_month = np.int64(request.form["arrival_month"])
        arrival_date = np.int64(request.form["arrival_date"])
        market_segment_type = np.int64(request.form["market_segment_type"])
        no_of_week_nights = np.int64(request.form["no_of_week_nights"])
        no_of_weekend_nights = np.int64(request.form["no_of_weekend_nights"])
        type_of_meal_plan = np.int64(request.form["type_of_meal_plan"])
        room_type_reserved = np.int64(request.form["room_type_reserved"])

        input_data = np.array([[
            lead_time, 
            no_of_special_requests, 
            avg_price_per_room, 
            arrival_month, 
            arrival_date, 
            market_segment_type, 
            no_of_week_nights, 
            no_of_weekend_nights, 
            type_of_meal_plan, 
            room_type_reserved
        ]])
        prediction = model.predict(input_data)
        prediction_probability = model.predict_proba(input_data)
        return render_template("index.html", prediction=prediction[0], prediction_probability=prediction_probability)
    return render_template("index.html", prediction=None, prediction_probability=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
