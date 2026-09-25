import streamlit as st
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id="RihabMhd/flight-price-model",
    filename="flight_price_model.joblib"
)

#configuration


st.set_page_config(
    page_title="Flight Price Predictor",
    layout="centered"
)



@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


#page title

st.title("✈️ Flight Price Predictor")

st.write(
    "Enter the characteristics of your flight to estimate its ticket price."
)


#input form

with st.form("flight_form"):

    st.subheader("Flight Information")

    col1, col2 = st.columns(2)

    with col1:

        airline = st.selectbox(
            "Airline",
            [
                "SpiceJet",
                "AirAsia",
                "Vistara",
                "GO_FIRST",
                "Indigo",
                "Air_India"
            ]
        )

        source_city = st.selectbox(
            "Departure City",
            [
                "Delhi",
                "Mumbai",
                "Bangalore",
                "Kolkata",
                "Hyderabad",
                "Chennai"
            ]
        )

        destination_city = st.selectbox(
            "Destination City",
            [
                "Mumbai",
                "Bangalore",
                "Kolkata",
                "Hyderabad",
                "Chennai",
                "Delhi"
            ]
        )

        departure_time = st.selectbox(
            "Departure Time",
            [
                "Evening",
                "Early_Morning",
                "Morning",
                "Afternoon",
                "Night",
                "Late_Night"
            ]
        )

    with col2:

        arrival_time = st.selectbox(
            "Arrival Time",
            [
                "Night",
                "Morning",
                "Early_Morning",
                "Afternoon",
                "Evening",
                "Late_Night"
            ]
        )

        stops = st.selectbox(
            "Number of Stops",
            [
                "zero",
                "one",
                "two_or_more"
            ]
        )

        flight_class = st.selectbox(
            "Class",
            [
                "Economy",
                "Business"
            ]
        )

        duration = st.number_input(
            "Duration (hours)",
            min_value=0.0,
            max_value=50.0,
            value=2.0,
            step=0.1
        )

        days_left = st.number_input(
            "Days Before Flight",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )

    submitted = st.form_submit_button(
        "Predict Price"
    )


#prediction

if submitted:

    input_data = pd.DataFrame({
        "airline": [airline],
        "source_city": [source_city],
        "departure_time": [departure_time],
        "stops": [stops],
        "arrival_time": [arrival_time],
        "destination_city": [destination_city],
        "class": [flight_class],
        "duration": [duration],
        "days_left": [days_left]
    })

    try:

        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")

        st.metric(
            label="Estimated Flight Price",
            value=f"₹ {prediction:,.2f}"
        )

        st.subheader("Flight Information")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(
            f"An error occurred during prediction: {e}"
        )