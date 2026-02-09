import streamlit as st
st.set_page_config(page_title="Decision Tree Classification", layout="centered")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Load CSS
# def load_css(file):
#     with open(file) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# load_css("style.css")

# Title
st.markdown("""
<div class="card">
    <h1>Decision Tree Classifier</h1>
    <p>Classify Houses into <b>High Value</b> or <b>Low Value</b></p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")

df = load_data()

# Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# Handle missing values
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)

# Feature Engineering
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# Create Classification Target
median_value = df["median_house_value"].median()
df["price_category"] = (df["median_house_value"] >= median_value).astype(int)

# Features & Target
X = df.drop(columns=["median_house_value", "ocean_proximity", "price_category"])
y = df["price_category"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pruning Control
depth = st.slider("Max Depth", 1, 20, 6)

model = DecisionTreeClassifier(
    max_depth=depth,
    min_samples_leaf=20,
    min_samples_split=40,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{accuracy*100:.2f}%")

# Confusion Matrix
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction Section
st.subheader("Make a Prediction")

longitude = st.slider("Longitude", float(df.longitude.min()), float(df.longitude.max()), float(df.longitude.mean()))
latitude = st.slider("Latitude", float(df.latitude.min()), float(df.latitude.max()), float(df.latitude.mean()))
housing_median_age = st.slider("Housing Median Age", int(df.housing_median_age.min()), int(df.housing_median_age.max()), int(df.housing_median_age.mean()))
total_rooms = st.slider("Total Rooms", int(df.total_rooms.min()), int(df.total_rooms.max()), int(df.total_rooms.mean()))
total_bedrooms = st.slider("Total Bedrooms", int(df.total_bedrooms.min()), int(df.total_bedrooms.max()), int(df.total_bedrooms.mean()))
population = st.slider("Population", int(df.population.min()), int(df.population.max()), int(df.population.mean()))
households = st.slider("Households", int(df.households.min()), int(df.households.max()), int(df.households.mean()))
median_income = st.slider("Median Income", float(df.median_income.min()), float(df.median_income.max()), float(df.median_income.mean()))

# Feature Engineering for Input
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households

input_df = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "rooms_per_household": rooms_per_household,
    "bedrooms_per_room": bedrooms_per_room,
    "population_per_household": population_per_household
}])

prediction = model.predict(input_df)[0]

result = "High Value House" if prediction == 1 else "Low Value House"

st.markdown(
    f"<div class='prediction-box'>Prediction: {result}</div>",
    unsafe_allow_html=True
)
