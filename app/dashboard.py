import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #10b981 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .subtitle-text {
        text-align: center;
        color: var(--text-color);
        opacity: 0.7;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(128, 128, 128, 0.4);
    }
    
    .metric-date {
        margin: 0;
        color: var(--text-color);
        opacity: 0.7;
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        margin: 10px 0;
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
    }
    
    .metric-status {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Pollutant small cards */
    .pollutant-box {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .pollutant-name {
        color: var(--text-color);
        opacity: 0.7;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 5px;
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


with st.spinner("Fetching high-resolution AI predictions..."):
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            data = response.json()
            forecast = data["forecast"]

            today_data = forecast[0]
            future_data = forecast[1:]

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
                                {"range": [0, 50], "color": "rgba(16, 185, 129, 0.15)"},
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
                st.markdown(
                    f"<p style='color: var(--text-color); opacity: 0.8;'>Temperature: <b>{today_data['temp']}°C</b> &nbsp;|&nbsp; Humidity: <b>{today_data['humidity']}%</b></p>",
                    unsafe_allow_html=True,
                )

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
            df = pd.DataFrame(forecast)

            fig = px.area(
                df,
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
