import streamlit as st
import pandas as pd
import json
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Wellness Tourism Package – Purchase Prediction",
    layout="wide"
)

@st.cache_resource
def load_model(hf_username):
    model_path = hf_hub_download(
        repo_id=f"{hf_username}/tourism-wellness-model",
        filename="best_tourism_wellness_model.joblib"
    )
    return joblib.load(model_path)

@st.cache_resource
def load_encoding_map(hf_username):
    map_path = hf_hub_download(
        repo_id=f"{hf_username}/tourism-package-prediction",
        repo_type="dataset",
        filename="encoding_map.json"
    )
    with open(map_path) as f:
        return json.load(f)

HF_USERNAME  = "sgnair"
model        = load_model(HF_USERNAME)
encoding_map = load_encoding_map(HF_USERNAME)

st.title("Wellness Tourism Package – Purchase Prediction")
st.markdown(
    "Enter customer details to generate a prediction of whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Demographics")
    age            = st.slider("Age", min_value=18, max_value=65, value=35)
    gender         = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
    occupation     = st.selectbox("Occupation", ["Free Lancer", "Large Business", "Salaried", "Small Business"])
    designation    = st.selectbox("Designation", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=22000, step=500)

with col2:
    st.subheader("Travel Profile")
    city_tier         = st.selectbox("City Tier", [1, 2, 3])
    type_of_contact   = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    passport          = st.selectbox("Passport Held", [0, 1], format_func=lambda x: "Yes" if x else "No")
    own_car           = st.selectbox("Car Owned", [0, 1], format_func=lambda x: "Yes" if x else "No")
    number_of_trips   = st.slider("Average Trips per Year", min_value=1, max_value=22, value=3)
    persons_visiting  = st.slider("Number of Persons Visiting", min_value=1, max_value=5, value=2)
    children_visiting = st.slider("Children (under 5) Visiting", min_value=0, max_value=3, value=1)
    preferred_star    = st.selectbox("Preferred Property Star Rating", [3, 4, 5])

with col3:
    st.subheader("Sales Interaction")
    product_pitched     = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
    pitch_satisfaction  = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    number_of_followups = st.slider("Number of Follow-ups", min_value=1, max_value=6, value=3)
    duration_of_pitch   = st.slider("Duration of Pitch (minutes)", min_value=5, max_value=127, value=20)

if st.button("Generate Prediction"):
    input_data = pd.DataFrame([{
        "Age":                      age,
        "TypeofContact":            encoding_map["TypeofContact"][type_of_contact],
        "CityTier":                 city_tier,
        "DurationOfPitch":          duration_of_pitch,
        "Occupation":               encoding_map["Occupation"][occupation],
        "Gender":                   encoding_map["Gender"][gender],
        "NumberOfPersonVisiting":   persons_visiting,
        "NumberOfFollowups":        number_of_followups,
        "ProductPitched":           encoding_map["ProductPitched"][product_pitched],
        "PreferredPropertyStar":    preferred_star,
        "MaritalStatus":            encoding_map["MaritalStatus"][marital_status],
        "NumberOfTrips":            number_of_trips,
        "Passport":                 passport,
        "PitchSatisfactionScore":   pitch_satisfaction,
        "OwnCar":                   own_car,
        "NumberOfChildrenVisiting": children_visiting,
        "Designation":              encoding_map["Designation"][designation],
        "MonthlyIncome":            monthly_income,
    }])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("Prediction: Likely to Purchase")
    else:
        st.error("Prediction: Unlikely to Purchase")

    st.metric(label="Estimated Purchase Probability", value=f"{probability:.1%}")
    st.progress(float(probability))
