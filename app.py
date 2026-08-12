import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# MODULE A: Data Engineering (Mock Data)
# ==========================================
@st.cache_data
def generate_timeseries_data(days=90):
    np.random.seed(42)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
    df = pd.DataFrame({'timestamp': timestamps})
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    base_load = 0.5 
    daily_seasonality = np.sin((df['hour'] - 6) * (2 * np.pi / 24)) + 1
    
    df['temperature'] = 20 + 5 * np.sin((df['hour'] - 8) * (2 * np.pi / 24)) + np.random.normal(0, 1, len(df))
    
    df['hvac_usage'] = np.where(df['temperature'] > 22, (df['temperature'] - 22) * 0.4, 0)
    df['hvac_usage'] += np.where(df['temperature'] < 18, (18 - df['temperature']) * 0.4, 0)
    
    df['lighting_usage'] = np.where((df['hour'] >= 18) & (df['hour'] <= 23), np.random.uniform(0.2, 0.8, len(df)), 0.1)
    df['appliance_usage'] = daily_seasonality * np.random.uniform(0.3, 1.2, len(df))
    
    df['total_kwh'] = base_load + df['hvac_usage'] + df['lighting_usage'] + df['appliance_usage']
    
    # Inject Anomalies
    anomaly_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df.loc[anomaly_indices, 'total_kwh'] *= np.random.uniform(2.5, 4.0, len(anomaly_indices))
    
    return df

@st.cache_data
def generate_spatial_room_data():
    rooms = [
        {'room': 'Living Room', 'x': 0, 'y': 0, 'z': 0, 'current_kwh': np.random.uniform(0.5, 2.0)},
        {'room': 'Kitchen', 'x': 5, 'y': 0, 'z': 0, 'current_kwh': np.random.uniform(1.0, 3.5)},
        {'room': 'Master Bedroom', 'x': 0, 'y': 5, 'z': 3, 'current_kwh': np.random.uniform(0.2, 1.0)},
        {'room': 'Guest Bedroom', 'x': 5, 'y': 5, 'z': 3, 'current_kwh': np.random.uniform(0.1, 0.5)},
        {'room': 'HVAC/Utility', 'x': 2.5, 'y': -3, 'z': 0, 'current_kwh': np.random.uniform(2.0, 5.0)},
        {'room': 'Garage', 'x': -5, 'y': 0, 'z': 0, 'current_kwh': np.random.uniform(0.1, 1.5)},
    ]
    return pd.DataFrame(rooms)


# ==========================================
# MODULE B: Machine Learning Engine
# ==========================================
@st.cache_data
def detect_anomalies(df):
    features = ['total_kwh', 'hvac_usage', 'temperature']
    X = df[features].fillna(0)
    
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    preds = iso_forest.fit_predict(X)
    
    df['is_anomaly'] = np.where(preds == -1, 1, 0)
    return df

@st.cache_data
def train_and_forecast(df, forecast_hours=24):
    X = df[['hour', 'day_of_week', 'temperature']].copy()
    y = df['total_kwh']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    last_timestamp = df['timestamp'].max()
    future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(forecast_hours)]
    
    future_df = pd.DataFrame({'timestamp': future_timestamps})
    future_df['hour'] = future_df['timestamp'].dt.hour
    future_df['day_of_week'] = future_df['timestamp'].dt.dayofweek
    future_df['temperature'] = 20 + 5 * np.sin((future_df['hour'] - 8) * (2 * np.pi / 24))
    
    X_future = future_df[['hour', 'day_of_week', 'temperature']]
    future_df['forecasted_kwh'] = model.predict(X_future)
    
    return future_df


# ==========================================
# MODULE C: NLP Recommendation Engine
# ==========================================
def generate_insights(df, anomalies):
    """
    Mocked LLM generation. In production, connect this to OpenAI/Gemini API via LangChain.
    """
    recent_anomalies = anomalies.tail(5)
    
    insights = []
    if not recent_anomalies.empty:
        insights.append(f"⚠️ **Alert:** We detected {len(recent_anomalies)} unusual energy spikes recently.")
        insights.append("💡 **Recommendation:** Your HVAC/Utility usage seems to be the primary driver of these spikes. Consider raising your thermostat by 2 degrees during peak hours (2 PM - 6 PM) to save approximately 12% on tomorrow's energy bill.")
        insights.append("🔍 **Diagnosis:** A sustained spike was detected when outside temperatures were normal, suggesting an appliance (like an oven or heater) may have been left on accidentally.")
    else:
        insights.append("✅ **All Good:** Your energy consumption is well within normal, efficient ranges.")
        
    return insights


# ==========================================
# MODULE D: Streamlit Dashboard UI
# ==========================================
st.set_page_config(page_title="AI Energy Analytics", layout="wide")

st.title("⚡ Residential Energy Analytics Platform")
st.markdown("AI-powered monitoring, forecasting, and spatial anomaly detection.")

# Load Data and Models
df_time = generate_timeseries_data(days=30)
df_time = detect_anomalies(df_time)
df_space = generate_spatial_room_data()
forecast_df = train_and_forecast(df_time, forecast_hours=48)

anomalies = df_time[df_time['is_anomaly'] == 1]

# Create UI Tabs
tab1, tab2, tab3 = st.tabs(["🏠 3D Home View", "📈 Forecasting & Monitoring", "🧠 AI Insights & Inefficiencies"])

with tab1:
    st.subheader("Interactive 3D Spatial Energy Map")
    st.markdown("Rotate and zoom to see which rooms are currently consuming the most energy.")
    
    # 3D Scatter plot using Plotly
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=df_space['x'],
        y=df_space['y'],
        z=df_space['z'],
        text=df_space['room'] + '<br>' + df_space['current_kwh'].round(2).astype(str) + ' kWh',
        mode='markers+text',
        textposition='top center',
        marker=dict(
            size=df_space['current_kwh'] * 20, # Scale bubble by energy use
            color=df_space['current_kwh'],     # Color by energy use
            colorscale='Inferno',
            opacity=0.8,
            colorbar=dict(title="Usage (kWh)")
        )
    )])
    
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='X (House Width)',
            yaxis_title='Y (House Length)',
            zaxis_title='Z (Floors)'
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.subheader("Next 48-Hour Consumption Forecast")
    
    # Plotting historical + forecast
    recent_history = df_time.tail(48)[['timestamp', 'total_kwh']].copy()
    recent_history['type'] = 'Historical'
    
    forecast_plot_df = forecast_df[['timestamp', 'forecasted_kwh']].copy()
    forecast_plot_df.rename(columns={'forecasted_kwh': 'total_kwh'}, inplace=True)
    forecast_plot_df['type'] = 'AI Forecast'
    
    combined_df = pd.concat([recent_history, forecast_plot_df])
    
    fig_line = px.line(combined_df, x='timestamp', y='total_kwh', color='type', 
                       title="Historical vs. Forecasted Energy Usage (kWh)",
                       color_discrete_map={"Historical": "blue", "AI Forecast": "orange"})
    
    st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    st.subheader("AI Recommendations & Anomaly Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Actionable Insights")
        insights = generate_insights(df_time, anomalies)
        for insight in insights:
            st.info(insight)
            
    with col2:
        st.markdown("### Detected Inefficiencies (Last 30 Days)")
        st.metric("Total Anomalies Flagged", len(anomalies))
        st.metric("Estimated Wasted Energy", f"{len(anomalies) * 2.5:.2f} kWh")
        
    st.markdown("### Anomaly Timeline")
    fig_anom = px.scatter(df_time.tail(200), x='timestamp', y='total_kwh', color='is_anomaly',
                          color_continuous_scale={0: 'blue', 1: 'red'},
                          title="Recent Usage with Highlighted Spikes (Red)")
    st.plotly_chart(fig_anom, use_container_width=True)
