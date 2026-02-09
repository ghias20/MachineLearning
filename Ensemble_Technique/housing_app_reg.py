import streamlit as st
st.set_page_config(page_title="Decision Tree", layout="centered")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")
st.markdown("""
<div class="card">
    <h1>Decision Tree Regressor</h1>
    <p>Predict Housing Prices using Decision Tree Regressor</p>
</div>
""", unsafe_allow_html=True)
@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")
df = load_data()
st.markdown('<div class="card" >',unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)
target_col = df.columns[-1]
target_col="median_house_value"
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

X = df.drop(columns=["median_house_value", "ocean_proximity"])
y= df["median_house_value"]
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
depth = st.slider("Max Depth", 1, 20, 5)
model = DecisionTreeRegressor(max_depth=depth,min_samples_leaf=20,min_samples_split=40,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("R2 Score", f"{r2_score(y_test, y_pred)*100:.2f}%")
st.subheader("Make a Prediction")
longitude = st.slider(
    "Longitude",
    float(df.longitude.min()),
    float(df.longitude.max()),
    float(df.longitude.mean())
)
latitude = st.slider(
    "Latitude",
    float(df.latitude.min()),
    float(df.latitude.max()),
    float(df.latitude.mean())
)
housing_median_age = st.slider(
    "Housing Median Age",
    int(df.housing_median_age.min()),
    int(df.housing_median_age.max()),
    int(df.housing_median_age.mean())
)
total_rooms = st.slider(
    "Total Rooms",
    int(df.total_rooms.min()),
    int(df.total_rooms.max()),
    int(df.total_rooms.mean())
)
total_bedrooms = st.slider(
    "Total Bedrooms",
    int(df.total_bedrooms.min()),
    int(df.total_bedrooms.max()),
    int(df.total_bedrooms.mean())
)
population = st.slider(
    "Population",
    int(df.population.min()),
    int(df.population.max()),
    int(df.population.mean())
)
households = st.slider(
    "Households",
    int(df.households.min()),
    int(df.households.max()),
    int(df.households.mean())
)

median_income = st.slider(
    "Median Income",
    float(df.median_income.min()),
    float(df.median_income.max()),
    float(df.median_income.mean())
)
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households
input_data = [[
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    rooms_per_household,
    bedrooms_per_room,
    population_per_household
]]

pred_price = model.predict(input_data)[0]

st.markdown(
    f"<div class='prediction-box'>Predicted Median House Value: ${pred_price:,.2f}</div>",
    unsafe_allow_html=True
)