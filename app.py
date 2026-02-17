import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import json
import os
import time

# ────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Surge Pricing Dashboard",
    page_icon="🚕",
    layout="wide",
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #ffcc00;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #ffcc00;
    }
    .stSlider > div > div > div > div {
        background-color: #ffcc00;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────
# DATA LOADING (CACHED)
# ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "output/models/surge_model.json"
    names_path = "output/models/feature_names.json"
    
    if not os.path.exists(model_path) or not os.path.exists(names_path):
        return None, None
    
    bst = xgb.Booster()
    bst.load_model(model_path)
    
    with open(names_path, "r") as f:
        feature_names = json.load(f)
        
    return bst, feature_names

@st.cache_data
def load_importance():
    path = "output/models/feature_importance.csv"
    if os.path.exists(path):
        return pd.read_csv(path).head(10)
    return None

# ────────────────────────────────────────────────────────────────────
# DASHBOARD UI
# ────────────────────────────────────────────────────────────────────
def main():
    st.title("🚕 NYC TLC Surge Pricing Engine")
    st.markdown("### Production-Grade Real-Time Surge Prediction")
    
    # ── Load Model ──
    model, feature_names = load_model()
    
    if model is None:
        st.error("❌ **Error:** Model files (`surge_model.json`, `feature_names.json`) not found in `output/models/`.")
        st.info("Please run the full pipeline (`python run_pipeline.py`) first to train the model.")
        return

    # ── Sidebar Inputs ──
    st.sidebar.header("🕹️ Simulation Controls")
    
    with st.sidebar:
        st.subheader("⏰ Temporal Features")
        hour = st.slider("Hour of Day", 0, 23, 14, help="24-hour format")
        day_of_week = st.selectbox(
            "Day of Week", 
            options=[0, 1, 2, 3, 4, 5, 6], 
            format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x]
        )
        
        st.subheader("📈 Demand & Supply")
        demand_60min = st.slider("Current Demand (Last 60m)", 1, 500, 150)
        supply_baseline = st.slider("Supply Baseline (Drivers)", 1, 200, 50)
        
        st.subheader("🌦️ Weather Conditions")
        temp = st.slider("Temperature (°F)", 0, 100, 72)
        precip = st.slider("Precipitation (Inches)", 0.0, 2.0, 0.0, step=0.1)
        rain_flag = 1 if precip > 0.1 else 0

    # ── Z-Score Logic (Mocking distribution from training data) ──
    # In production, these mean/std values would be saved in metrics.json
    mean_demand = 120.0
    std_demand = 40.0
    demand_zscore = (demand_60min - mean_demand) / std_demand

    # ── Prepare Feature Vector ──
    # Initialize all columns with 0
    input_dict = {name: 0.0 for name in feature_names}
    
    # Map user inputs
    input_dict.update({
        "hour": hour,
        "day_of_week": day_of_week,
        "demand_60min": demand_60min,
        "supply_baseline": supply_baseline,
        "demand_zscore": demand_zscore,
        "temperature_f": temp,
        "precipitation_in": precip,
        "weather_is_rain": rain_flag,
        "is_weekend": 1 if day_of_week >= 5 else 0,
        "is_rush_hour": 1 if (7 <= hour <= 10 or 16 <= hour <= 20) else 0,
    })

    # Create DataFrame aligned with training features
    input_df = pd.DataFrame([input_dict])[feature_names]
    dmatrix = xgb.DMatrix(input_df)

    # ── Prediction ──
    surge_prediction = model.predict(dmatrix)[0]
    surge_prediction = max(1.0, float(surge_prediction)) # Floor at 1.0x

    # ── Main Content Layout ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style='margin:0; color:#888;'>PREDICTED SURGE</h4>
            <h1 style='margin:0; font-size: 5rem; color:#ffcc00;'>{surge_prediction:.2f}x</h1>
            <p style='margin:0; color:#00ff00;'>{"🟢 Low Demand" if surge_prediction < 1.2 else "🔴 High Demand" if surge_prediction > 2.0 else "🟡 Moderate"}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = surge_prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Surge Multiplier Intensity", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 5.0], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#ffcc00"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [1.0, 1.5], 'color': 'rgba(0, 255, 0, 0.2)'},
                    {'range': [1.5, 2.5], 'color': 'rgba(255, 255, 0, 0.2)'},
                    {'range': [2.5, 5.0], 'color': 'rgba(255, 0, 0, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 4.5
                }
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Model Insights")
        importance_df = load_importance()
        
        if importance_df is not None:
            # Feature Importance Bar Chart
            fig_imp = go.Figure(go.Bar(
                x=importance_df['importance_gain'],
                y=importance_df['feature'],
                orientation='h',
                marker_color='#ffcc00'
            ))
            fig_imp.update_layout(
                title="Top Features (Instructional Gain)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white"},
                yaxis={'autorange': "reversed"}
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("⚠️ Feature importance data missing.")

        # Real-time state table
        st.subheader("📝 Input State")
        st.table(pd.DataFrame({
            "Feature": ["Demand Z-Score", "Rush Hour", "Weekend", "Is Raining"],
            "Value": [f"{demand_zscore:.2f}", "Yes" if input_dict["is_rush_hour"] else "No", 
                      "Yes" if input_dict["is_weekend"] else "No", "Yes" if rain_flag else "No"]
        }))

    st.markdown("---")
    st.caption("NYC Surge Pricing Engine v1.0.0 | Memory-Safe XGBoost Implementation")

if __name__ == "__main__":
    main()
