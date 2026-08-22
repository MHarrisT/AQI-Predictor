import os
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
        "aqi": aqi,
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

    return df


def store_features(df: pd.DataFrame):
    """Stores these features in the Feature store."""
    os.makedirs("/tmp", exist_ok=True)

    # Connect to Hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    # Version 3: Disabled online store to bypass Kafka authorization errors
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=3,
        description="Air Quality Index and Weather Features",
        primary_key=["city"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="HUDI",
    )

    # Insert the dataframe
    aqi_fg.insert(df, write_options={"wait_for_job": True})
    print("Successfully loaded features into Hopsworks!")


if __name__ == "__main__":
    print("Fetching raw data...")
    raw_df = fetch_raw_data()

    print("Computing features...")
    processed_df = compute_features(raw_df)

    print("Storing in Hopsworks Feature Store...")
    store_features(processed_df)
