import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

# Dashboard UI Config
st.set_page_config(page_title="AQI Predictor", page_icon="🌬️", layout="wide")

# Adaptive Custom CSS using Streamlit's native variables
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Gradient Title (Emerald to Teal - fresh air vibe) */
    .title-text {
        background: linear-gradient(90deg, #10B981, #0D9488);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0px;
    }
    
    .subtitle-text {
        color: var(--text-color);
        opacity: 0.8;
        font-size: 1.2rem;
        margin-top: -10px;
        margin-bottom: 40px;
    }
    
    /* Minimalist metric cards */
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border: 1px solid #10B981;
    }

    .metric-date {
        color: var(--text-color);
        opacity: 0.7;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 10px 0px;
    }

    .metric-status {
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Clean pollutant display */
    .pollutant-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px;
        border-bottom: 1px solid rgba(128,128,128,0.1);
    }
    
    .pollutant-name {
        color: var(--text-color);
        opacity: 0.8;
        font-weight: 500;
    }
    
    .pollutant-val {
        color: var(--text-color);
        font-size: 1.2rem;
        font-weight: 700;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 class='title-text'>🌬️ Lahore AQI Predictor</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p class='subtitle-text'>Real-time AI Air Quality forecasting powered by XGBoost & Hopsworks</p>",
    unsafe_allow_html=True,
)

API_URL = "http://localhost:8000/predict"


def get_aqi_details(aqi):
    if aqi <= 50:
        return "#10B981", "Good"
    elif aqi <= 100:
        return "#F59E0B", "Moderate"
    elif aqi <= 150:
        return "#F97316", "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "#EF4444", "Unhealthy"
    elif aqi <= 300:
        return "#D946EF", "Very Unhealthy"
    else:
        return "#9F1239", "Hazardous"


tab1, tab2, tab3 = st.tabs(
    ["Forecast & Alerts", "Exploratory Data Analysis (EDA)", "Model Evaluation"]
)

with tab1:
    with st.spinner("Fetching high-resolution AI predictions..."):
        try:
            response = requests.get(API_URL)
            if response.status_code == 200:
                data = response.json()
                forecast = data["forecast"]

                today_data = forecast[0]
                future_data = forecast[1:]

                # Hazardous Alert
                if today_data["predicted_aqi"] > 300:
                    st.error(
                        "🚨 **HAZARDOUS AIR QUALITY WARNING** 🚨\n\nThe predicted AQI has exceeded 300. It is strongly advised to stay indoors and keep windows closed."
                    )

                st.markdown(
                    "### 📍 Current Forecast Details (Today)", unsafe_allow_html=True
                )

                col1, col2 = st.columns([1, 1.5])

                with col1:
                    # Gauge Chart for Today
                    aqi = today_data["predicted_aqi"]
                    color, status = get_aqi_details(aqi)

                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=aqi,
                            title={"text": "Predicted AQI", "font": {"size": 24}},
                            number={
                                "font": {"size": 60, "color": color},
                                "valueformat": ".0f",
                            },
                            gauge={
                                "axis": {
                                    "range": [0, 500],
                                    "tickwidth": 1,
                                    "tickcolor": "gray",
                                },
                                "bar": {"color": color},
                                "bgcolor": "rgba(128,128,128,0.1)",
                                "borderwidth": 0,
                                "steps": [
                                    {
                                        "range": [0, 50],
                                        "color": "rgba(16, 185, 129, 0.15)",
                                    },
                                    {
                                        "range": [51, 100],
                                        "color": "rgba(245, 158, 11, 0.15)",
                                    },
                                    {
                                        "range": [101, 150],
                                        "color": "rgba(249, 115, 22, 0.15)",
                                    },
                                    {
                                        "range": [151, 200],
                                        "color": "rgba(239, 68, 68, 0.15)",
                                    },
                                    {
                                        "range": [201, 300],
                                        "color": "rgba(217, 70, 239, 0.15)",
                                    },
                                    {
                                        "range": [301, 500],
                                        "color": "rgba(159, 18, 57, 0.15)",
                                    },
                                ],
                            },
                        )
                    )
                    fig_gauge.update_layout(
                        font={"family": "Outfit"}, height=350, margin=dict(t=50, b=0)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    st.markdown(
                        f"<h3 style='text-align: center; color: {color}; margin-top: -30px;'>{status}</h3>",
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### Weather & Pollutant Breakdown")

                    p_cols = st.columns(3)
                    pollutants = [
                        ("PM2.5", today_data["pm2_5"], "µg/m³"),
                        ("PM10", today_data["pm10"], "µg/m³"),
                        ("Ozone (O3)", today_data["o3"], "µg/m³"),
                        ("Nitrogen Dioxide (NO2)", today_data["no2"], "µg/m³"),
                        ("Carbon Monoxide (CO)", today_data["co"], "µg/m³"),
                    ]

                    for idx, (p_name, p_val, p_unit) in enumerate(pollutants):
                        with p_cols[idx % 3]:
                            st.markdown(
                                f"""
                                <div class="pollutant-box">
                                    <div class="pollutant-name">{p_name}</div>
                                    <div class="pollutant-val">{p_val:.1f} <span style="font-size:0.7rem; opacity: 0.7;">{p_unit}</span></div>
                                </div>
                                <br>
                            """,
                                unsafe_allow_html=True,
                            )

                st.divider()

                # Future Forecasts Area
                st.markdown("### 📅 Upcoming 3-Day Outlook")

                f_cols = st.columns(3)
                for i, day_data in enumerate(future_data):
                    aqi = day_data["predicted_aqi"]
                    color, status = get_aqi_details(aqi)

                    date_obj = pd.to_datetime(day_data["date"])
                    day_name = "Tomorrow" if i == 0 else date_obj.strftime("%A, %b %d")

                    with f_cols[i]:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <p class="metric-date">{day_name}</p>
                                <h1 class="metric-value" style="color: {color};">{aqi:.0f}</h1>
                                <p class="metric-status" style="color: {color};">{status}</p>
                                <div style="margin-top: 15px; border-top: 1px solid rgba(128,128,128,0.2); padding-top: 10px;">
                                    <span style="color: var(--text-color); opacity: 0.7; font-size: 0.9rem;">Temp: {day_data["temp"]}°C | Hum: {day_data["humidity"]}%</span><br>
                                    <span style="color: var(--text-color); opacity: 0.7; font-size: 0.85rem;">PM2.5: {day_data["pm2_5"]} | PM10: {day_data["pm10"]}</span>
                                </div>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )

                # Chart
                st.markdown(
                    "<br><h3 style='font-weight: 600;'>📈 4-Day Forecast Trend</h3>",
                    unsafe_allow_html=True,
                )
                df_forecast = pd.DataFrame(forecast)

                fig = px.area(
                    df_forecast,
                    x="date",
                    y="predicted_aqi",
                    markers=True,
                    labels={
                        "date": "Forecast Date",
                        "predicted_aqi": "Predicted AQI Score",
                    },
                )

                fig.update_traces(
                    line_color="#06b6d4",
                    fillcolor="rgba(6, 182, 212, 0.15)",
                    line_width=4,
                    marker=dict(
                        size=12, color="#10b981", line=dict(width=2, color="white")
                    ),
                )

                fig.update_layout(
                    xaxis=dict(showgrid=False, title_font=dict(size=14)),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(128,128,128,0.2)",
                        title_font=dict(size=14),
                    ),
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=350,
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(
                    f"Failed to fetch data from API. Status Code: {response.status_code}"
                )
        except requests.exceptions.ConnectionError:
            st.error(
                "Backend API is not running. Please start the FastAPI server with `python -m uvicorn app.api:app --reload`."
            )

with tab2:
    st.markdown("### 📊 Exploratory Data Analysis")
    st.markdown(
        "Explore the underlying relationships in the historical air quality dataset."
    )

    with st.spinner("Loading Exploratory Data Analysis..."):
        try:
            eda_resp = requests.get(f"{API_URL.replace('/predict', '/eda-data')}")
            if eda_resp.status_code == 200:
                eda_df = pd.DataFrame(eda_resp.json())

                st.markdown("#### PM2.5 vs AQI Trend")
                fig_scatter = px.scatter(
                    eda_df,
                    x="pm2_5",
                    y="aqi",
                    color="aqi",
                    color_continuous_scale="magma",
                    labels={"pm2_5": "PM2.5 (µg/m³)", "aqi": "AQI Score"},
                    title="Relationship between Particulate Matter and AQI",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                st.markdown("#### Feature Correlation Matrix")
                st.image(
                    f"{API_URL.replace('/predict', '/analytics-image/correlation_matrix.png')}",
                    use_container_width=True,
                )

            else:
                st.warning(
                    "EDA data not available yet. Has the model finished training?"
                )
        except Exception as e:
            st.warning(f"Backend API is not reachable to fetch EDA data. Error: {e}")

with tab3:
    st.markdown("### 🤖 AI Model Evaluation")

    with st.spinner("Loading Model Metrics..."):
        try:
            metrics_resp = requests.get(f"{API_URL.replace('/predict', '/models')}")
            if metrics_resp.status_code == 200:
                metrics_data = metrics_resp.json()

                st.markdown("#### 🏆 Model Leaderboard")
                # Convert nested JSON into a DataFrame
                table_data = []
                for model_name, data in metrics_data.items():
                    table_data.append(
                        {
                            "Model": model_name,
                            "Train R²": data["train"]["R2"],
                            "Test R²": data["test"]["R2"],
                            "Test MAE": data["test"]["MAE"],
                            "Test RMSE": data["test"]["RMSE"],
                        }
                    )
                metrics_df = pd.DataFrame(table_data).sort_values(
                    "Test R²", ascending=False
                )
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                st.markdown("#### 🧠 Feature Importance (SHAP)")
                st.markdown(
                    "This chart explains *why* the XGBoost model makes its predictions by showing the impact of each feature."
                )
                st.image(
                    f"{API_URL.replace('/predict', '/analytics-image/shap_summary.png')}",
                    use_container_width=True,
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🎯 Prediction Error by Category")
                    st.image(
                        f"{API_URL.replace('/predict', '/analytics-image/error_by_category.png')}",
                        use_container_width=True,
                    )
                with col2:
                    st.markdown("#### 📉 Residual Distribution")
                    st.image(
                        f"{API_URL.replace('/predict', '/analytics-image/residuals.png')}",
                        use_container_width=True,
                    )

            else:
                st.warning("Model evaluation metrics not available yet.")
        except Exception as e:
            st.warning(
                f"Backend API is not reachable to fetch model metrics. Error: {e}"
            )
