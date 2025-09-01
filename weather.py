import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
from datetime import datetime

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned_Indian_earthquake_data.csv', parse_dates=['Date'])
    df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
    return df

df = load_data()

st.title('Earthquake Visualization Dashboard')

# Sidebar for filtering
st.sidebar.header('Filters')
min_magnitude = st.sidebar.slider('Minimum Magnitude', float(df['Magnitude'].min()), float(df['Magnitude'].max()), float(df['Magnitude'].min()))
date_range = st.sidebar.date_input('Date Range', [df['Date'].min(), df['Date'].max()])

# Filter the dataframe
filtered_df = df[(df['Magnitude'] >= min_magnitude) & 
                 (df['Date'] >= pd.Timestamp(date_range[0])) & 
                 (df['Date'] <= pd.Timestamp(date_range[1]))]

# Map
st.subheader('Earthquake Locations')
m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)

for idx, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=row['Magnitude'] * 2,
        color='red' if row['Magnitude'] >= 4 else 'orange',
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['Location']}, Magnitude: {row['Magnitude']}, Date: {row['Date']}",
    ).add_to(m)

folium_static(m)

# Magnitude Distribution
st.subheader('Magnitude Distribution')
fig_mag = px.histogram(filtered_df, x='Magnitude', nbins=20, title='Distribution of Earthquake Magnitudes')
st.plotly_chart(fig_mag)

# Earthquakes over time
st.subheader('Earthquakes Over Time')
fig_time = px.scatter(filtered_df, x='Date', y='Magnitude', hover_data=['Location'], 
                      title='Earthquake Magnitudes Over Time')
st.plotly_chart(fig_time)

# Top 10 strongest earthquakes
st.subheader('Top 10 Strongest Earthquakes')
top_10 = filtered_df.nlargest(10, 'Magnitude')
st.table(top_10[['Date', 'Magnitude', 'Location']])

# Download filtered data
st.subheader('Download Filtered Data')
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name="filtered_earthquake_data.csv",
    mime="text/csv",
)