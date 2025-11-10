import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from us_states import state_name_map, state_centroids

print(state_name_map)
st.set_page_config(layout="wide", page_title="Rent vs Buy Forecast Dashboard")

# ---------- Utility Functions ----------
@st.cache_data(ttl=3600)
def load_data(path_options):
    for p in path_options:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            elif p.suffix == ".csv":
                return pd.read_csv(p)
    return None

def ensure_datetime(df):
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

def prepare_dataframe(df, rent_col, mortgage_col):
    df = df.copy()
    df = ensure_datetime(df)
    df["Rent_to_Mortgage_Ratio"] = df[rent_col] / df[mortgage_col]
    df["Monthly_Savings_or_Cost"] = df[mortgage_col] - df[rent_col]
    df["Affordability_Index"] = df[mortgage_col] / (df[rent_col] + df[mortgage_col])
    df["Decision"] = np.where(
        (df["Rent_to_Mortgage_Ratio"] > 1.1),
        "Buy",
        np.where(df["Rent_to_Mortgage_Ratio"] < 0.9, "Rent", "Neutral")
    )
    return df

def plot_state_map(df_state, title_suffix):
    fig = go.Figure(data=go.Choropleth(
        locations=df_state['StateName'],
        z=df_state['Rent_to_Mortgage_Ratio'],
        locationmode='USA-states',
        colorscale='RdYlGn',
        colorbar_title='Buy-Rent Index',
        hoverinfo='text',
        text=[f"{state_name_map[row.StateName]}<br>Buy-Rent Index: {row.Rent_to_Mortgage_Ratio:.2f}" # Updated hover text
          for index, row in df_state.iterrows()]
    ))
    fig.add_trace(
      go.Scattergeo(
          locationmode='USA-states',
          locations=df_state['StateName'],
          lat=[state_centroids[state][0] for state in df_state['StateName']],
          lon=[state_centroids[state][1] for state in df_state['StateName']],
          mode='text',
          text=df_state['StateName'], # Display state abbreviation
          textfont=dict(
              size=10,
              color="black"
          ),
          showlegend=False,
          hoverinfo='skip' # Skip hover info for the text labels
      )
    )
    fig.update_layout(
        title_text=f'Buy vs Rent Favorability by State (Buy = Green, Rent = Red)',
        geo_scope='usa',
        width=900,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def plot_top10_bar(df, value_col, title):
    st.subheader(f"Top 10 by {title}")
    top10 = df.groupby('RegionName')[value_col].mean().sort_values(ascending=False).head(10).reset_index()
    top10.columns = ['Metro Area', 'Average Value']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x='Average Value', y='Metro Area', data=top10, palette="viridis", ax=ax)
    for index, row in top10.iterrows():
        ax.text(row['Average Value'], index, f"${row['Average Value']:,.0f}",
                color='black', ha="left", va="center", fontsize=10)
    ax.set_xlabel(f"{value_col} (USD)", fontsize=12)
    ax.set_ylabel("")
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.grid(axis='y', visible=False)
    st.pyplot(fig)

# ---------- Load Data ----------
forecast_path_options = [
    Path("data/final_forecasts.parquet"),
    Path("data/final_forecasts.csv"),
    Path("final_forecasts.parquet"),
    Path("final_forecasts.csv")
]
current_path_options = [
    Path("data_processed/current.parquet"),
    Path("data_processed/current.csv"),
    Path("current_data.parquet"),
    Path("current_data.csv")
]

df_forecast = load_data(forecast_path_options)
df_current = load_data(current_path_options)

if df_forecast is None or df_current is None:
    st.error("Missing datasets. Please ensure both current and forecast files are in place.")
    st.stop()

# ---------- Sidebar ----------
st.sidebar.title("Data Mode")
data_mode = st.sidebar.radio("Select Data Source:", ["Current", "Forecasted"])

if data_mode == "Current":
    df = prepare_dataframe(df_current, "RentValue", "MortgageValue")
    rent_col, mortgage_col = "RentValue", "MortgageValue"
    title_suffix = "Current Market Data"
else:
    df = prepare_dataframe(df_forecast, "ForecastedRentValue", "ForecastedMortgageValue")
    rent_col, mortgage_col = "ForecastedRentValue", "ForecastedMortgageValue"
    title_suffix = "Forecasted Data"

# ---------- KPIs ----------
st.title("🏡 Rent vs Buy Decision Forecasting Dashboard")
st.markdown(f"Displaying **{title_suffix}** across U.S. metros.")

avg_rent = df[rent_col].mean()
avg_mortgage = df[mortgage_col].mean()
overall_ratio = df["Rent_to_Mortgage_Ratio"].mean()
buy_share = (df["Decision"] == "Buy").mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Rent", f"${avg_rent:,.0f}")
col2.metric("Avg Mortgage", f"${avg_mortgage:,.0f}")
col3.metric("Avg Rent-to-Mortgage Ratio", f"{overall_ratio:.2f}")
col4.metric("% Regions favoring Buy", f"{buy_share*100:.1f}%")

st.markdown("---")

# ---------- Map ----------
st.subheader(f"Buy vs Rent Favorability Map ({title_suffix})")
df_state = df.groupby('StateName')['Rent_to_Mortgage_Ratio'].mean().reset_index()
fig = plot_state_map(df_state, title_suffix)
st.plotly_chart(fig, use_container_width=True)

plot_top10_bar(df, rent_col, "Average Rent Value")
plot_top10_bar(df, mortgage_col, "Average Mortgage Value")
