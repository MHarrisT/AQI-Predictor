import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import hopsworks

# Initialize environment configurations
load_dotenv()
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

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


def fetch_historical_pollution(days=365) -> pd.DataFrame:
    """Fetches historical pollution data from the OpenWeather API."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())

    # OpenWeather provides free historical air pollution data!
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_unix}&end={end_unix}&appid={WEATHER_API_KEY}"

    response = requests.get(url).json()

    records = []
    if "list" in response:
        for item in response["list"]:
            comp = item["components"]
            records.append(
                {
                    "city": CITY_NAME,
                    "timestamp": datetime.fromtimestamp(item["dt"], tz=timezone.utc),
                    "aqi": calculate_epa_aqi(comp.get("pm2_5", 0)),
                    "co": comp.get("co"),
                    "no2": comp.get("no2"),
                    "o3": comp.get("o3"),
                    "pm2_5": comp.get("pm2_5"),
                    "pm10": comp.get("pm10"),
                    # Note: Free tier historical weather data requires a different paid API.
                    # We will use static placeholders for temp/humidity in the historical dataset
                    # so the baseline model can focus purely on pollution trends.
                    "temp": 25.0,
                    "humidity": 50,
                }
            )
    return pd.DataFrame(records)


def compute_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes time-based and rolling features for the historical dataset."""
    # Ensure chronological order for rolling calculations
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    # Engineered Features (Now we can calculate these properly with historical context!)
    df["lag_1_aqi"] = df["aqi"].shift(1)

    # Calculate change rate, handling potential division by zero
    df["aqi_change_rate"] = df.apply(
        lambda row: (
            (row["aqi"] - row["lag_1_aqi"]) / row["lag_1_aqi"]
            if row["lag_1_aqi"] > 0
            else 0
        ),
        axis=1,
    )

    # 7-day rolling average (24 hours * 7 days)
    df["rolling_avg_7_day"] = df["aqi"].rolling(window=24 * 7, min_periods=1).mean()

    # Drop the first row since it will have NaN values from the shift() operation
    df = df.dropna()

    return df


def store_historical_features(df: pd.DataFrame):
    """Pushes the backfill dataset to Hopsworks."""
    os.makedirs("/tmp", exist_ok=True)

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    # Bump to version 4 for the 0-500 scale
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=4,
        description="Air Quality Index and Weather Features",
        primary_key=["city"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="HUDI",
    )

    print(f"Uploading {len(df)} historical rows to Hopsworks...")
    aqi_fg.insert(df, write_options={"wait_for_job": True})
    print("Backfill complete!")


if __name__ == "__main__":
    print("Fetching historical data (this may take a moment)...")
    raw_df = fetch_historical_pollution(days=365)

    if raw_df.empty:
        print("No historical data found. Check your API key or coordinates.")
    else:
        print("Computing rolling features...")
        processed_df = compute_historical_features(raw_df)

        print("Storing backfill data in Hopsworks...")
        store_historical_features(processed_df)
