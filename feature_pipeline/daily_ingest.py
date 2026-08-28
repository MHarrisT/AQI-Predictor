import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import hopsworks

# Initialize environment configurations
load_dotenv()
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Target City Coordinates for Lahore
LAT, LON = 31.5204, 74.3587
CITY_NAME = "Lahore"


def calculate_epa_aqi(pm25):
    if pm25 is None:
        return 0
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low)
    return 500  # Max out if very high


def fetch_raw_data() -> pd.DataFrame:
    """Fetches raw weather and pollutant data from the OpenWeather API."""
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={WEATHER_API_KEY}"
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={WEATHER_API_KEY}&units=metric"

    poll_resp = requests.get(pollution_url).json()
    weath_resp = requests.get(weather_url).json()

    components = poll_resp["list"][0]["components"]
    aqi = poll_resp["list"][0]["main"]["aqi"]
    weather_main = weath_resp["main"]

    data = {
        "city": CITY_NAME,
        "timestamp": datetime.now(timezone.utc),
        "aqi": calculate_epa_aqi(components.get("pm2_5", 0)),
        "co": components.get("co"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "temp": weather_main.get("temp"),
        "humidity": weather_main.get("humidity"),
    }
    return pd.DataFrame([data])


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes time-based and derived features[cite: 1]."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract time-based features (hour, day, month)[cite: 1]
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    # Placeholders for derived features like AQI change rate[cite: 1]
    # These will be computed properly during the backfill process
    df["lag_1_aqi"] = 0.0
    df["aqi_change_rate"] = 0.0
    df["rolling_avg_7_day"] = 0.0

    # Enforce float types to match the expected 'double' schema in Hopsworks
    float_cols = ["co", "no2", "o3", "pm2_5", "pm10", "temp"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def store_features(df: pd.DataFrame):
    """Stores these features in the Feature store."""
    os.makedirs("/tmp", exist_ok=True)

    # Connect to Hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    # Version 4: 0-500 EPA AQI scale
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=4,
        description="Air Quality Index and Weather Features",
        primary_key=["city"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="HUDI",
    )

    # Insert the dataframe with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            aqi_fg.insert(df, write_options={"wait_for_job": False})
            break
        except Exception as e:
            print(f"Insert attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(10)
            
    print("Successfully loaded features into Hopsworks!")


if __name__ == "__main__":
    print("Fetching raw data...")
    raw_df = fetch_raw_data()

    print("Computing features...")
    processed_df = compute_features(raw_df)

    print("Storing in Hopsworks Feature Store...")
    store_features(processed_df)
