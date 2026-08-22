import os
import joblib
import pandas as pd
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI
from fastapi.responses import FileResponse
import json
import hopsworks
import requests
from dotenv import load_dotenv

# Load env variables
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT, LON = 31.5204, 74.3587
CITY_NAME = "Lahore"

app = FastAPI(title="AQI Predictor API")

model = None
scaler = None
model_dir_path = ""


@app.on_event("startup")
def load_model():
    global model, scaler, model_dir_path
    print("Connecting to Hopsworks to download model...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()

    # Download the best model (which we named aqi_predictor_best)
    models = mr.get_models("aqi_predictor_best")
    latest_version = max([m.version for m in models])
    hw_model = mr.get_model("aqi_predictor_best", version=latest_version)
    import shutil
    import os
    
    local_model_dir = os.path.join(os.getcwd(), "model_dir")
    if os.path.exists(local_model_dir):
        shutil.rmtree(local_model_dir)
        
    model_dir_path = hw_model.download(local_path=local_model_dir)

    if os.path.exists(model_dir_path + "/aqi_model.json"):
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(model_dir_path + "/aqi_model.json")
    elif os.path.exists(model_dir_path + "/aqi_model.h5"):
        import tensorflow as tf
        model = tf.keras.models.load_model(model_dir_path + "/aqi_model.h5")
    else:
        model = joblib.load(model_dir_path + "/aqi_model.pkl")
    scaler = joblib.load(model_dir_path + "/scaler.pkl")
    print("Model and scaler loaded successfully.")


@app.get("/predict")
def predict_3_days():
    """Predict AQI for the next 3 days using OpenWeather forecasts and Hopsworks Model."""
    # 1. Fetch OpenWeather forecasts (5 day / 3 hour)
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}"
    weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric"

    poll_resp = requests.get(pollution_url).json()
    weath_resp = requests.get(weather_url).json()

    predictions = []

    now = datetime.now(timezone.utc)
    target_dates = [now + timedelta(days=i) for i in range(0, 4)]

    for target_date in target_dates:
        # Find closest forecast entry for this target date (around noon)
        target_noon = target_date.replace(hour=12, minute=0, second=0, microsecond=0)
        target_timestamp = int(target_noon.timestamp())

        # Find closest weather forecast
        w_entry = min(weath_resp["list"], key=lambda x: abs(x["dt"] - target_timestamp))
        # Find closest pollution forecast
        p_entry = min(poll_resp["list"], key=lambda x: abs(x["dt"] - target_timestamp))

        comps = p_entry["components"]

        # Build features dataframe
        df = pd.DataFrame(
            [
                {
                    "co": comps.get("co"),
                    "no2": comps.get("no2"),
                    "o3": comps.get("o3"),
                    "pm2_5": comps.get("pm2_5"),
                    "temp": w_entry["main"].get("temp"),
                    "humidity": w_entry["main"].get("humidity"),
                    "hour": target_noon.hour,
                    "day": target_noon.day,
                    "month": target_noon.month,
                    "lag_1_aqi": p_entry["main"][
                        "aqi"
                    ],  # Use forecasted AQI index as lag approximation
                    "aqi_change_rate": 0.0,  # Naive approximation
                    "rolling_avg_7_day": p_entry["main"]["aqi"],  # Naive approximation
                }
            ]
        )

        # Scale features
        scaled_features = scaler.transform(df)

        # Predict using XGBoost
        predicted_aqi = model.predict(scaled_features)[0]

        predictions.append(
            {
                "date": target_noon.strftime("%Y-%m-%d"),
                "predicted_aqi": float(predicted_aqi),
                "temp": w_entry["main"].get("temp"),
                "humidity": w_entry["main"].get("humidity"),
                "co": comps.get("co"),
                "no2": comps.get("no2"),
                "o3": comps.get("o3"),
                "pm2_5": comps.get("pm2_5"),
                "pm10": comps.get("pm10"),
            }
        )

    return {"city": CITY_NAME, "forecast": predictions}


@app.get("/models")
def get_models():
    """Returns model metrics JSON."""
    with open(f"{model_dir_path}/model_metrics.json", "r") as f:
        return json.load(f)


@app.get("/eda-data")
def get_eda():
    """Returns EDA sample CSV as JSON."""
    df = pd.read_csv(f"{model_dir_path}/eda_sample.csv")
    return df.to_dict(orient="records")


@app.get("/analytics-image/{image_name}")
def get_analytics_image(image_name: str):
    """Serves the generated analytics plots."""
    return FileResponse(f"{model_dir_path}/{image_name}")
