import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# Page Config
st.set_page_config(
    page_title="FuturePath - Smart Career Prediction System",
    page_icon="🔮",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    try:
        if os.path.exists("career_model_pipeline.pkl"):
            return joblib.load("career_model_pipeline.pkl")

        elif os.path.exists("career_model_pipeline.pkl.xz"):
            return joblib.load("career_model_pipeline.pkl.xz")

        else:
            return None

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_data = load_model()

if model_data is None:
    st.error("Model file not found!")
    st.stop()

# Extract model data
pipeline = model_data["pipeline"]
metadata = model_data["metadata"]
target_classes = model_data["target_classes"]

st.title("🔮 FuturePath AI")
st.subheader("Career Recommendation System")

with st.form("prediction_form"):

    st.header("🧠 Skills")

    coding_rating = st.slider("Coding Skills Rating", 1, 9, 5)
    logical_rating = st.slider("Logical Quotient Rating", 1, 9, 5)
    hackathons = st.slider("Hackathons Participated", 0, 6, 1)
    public_speaking = st.slider("Public Speaking Points", 1, 9, 5)

    st.header("📚 Preferences")

    sub = st.selectbox(
        "Interested Subjects",
        metadata["Interested subjects"]
    )

    cert = st.selectbox(
        "Certifications",
        metadata["certifications"]
    )

    workshop = st.selectbox(
        "Workshops",
        metadata["workshops"]
    )

    extra_courses = st.selectbox(
        "Extra Courses",
        metadata["Extra-courses did"]
    )

    read_write = st.selectbox(
        "Reading and Writing Skills",
        metadata["reading and writing skills"]
    )

    memory = st.selectbox(
        "Memory Capability",
        metadata["memory capability score"]
    )

    self_learning = st.selectbox(
        "Self Learning Capability",
        metadata["self-learning capability?"]
    )

    introvert = st.selectbox(
        "Introvert",
        metadata["Introvert"]
    )

    teams = st.selectbox(
        "Worked in Teams Ever?",
        metadata["worked in teams ever?"]
    )

    hard_smart = st.selectbox(
        "Hard/Smart Worker",
        metadata["hard/smart worker"]
    )

    career_area = st.selectbox(
        "Interested Career Area",
        metadata["interested career area"]
    )

    company_type = st.selectbox(
        "Type of Company Want To Settle In?",
        metadata["Type of company want to settle in?"]
    )

    mgmt_tech = st.selectbox(
        "Management or Technical",
        metadata["Management or Technical"]
    )

    books = st.selectbox(
        "Interested Type of Books",
        metadata["Interested Type of Books"]
    )

    senior_input = st.selectbox(
        "Taken Inputs From Seniors",
        metadata["Taken inputs from seniors or elders"]
    )

    submit = st.form_submit_button("Predict Career")

if submit:

    user_inputs = {
        "Logical quotient rating": logical_rating,
        "hackathons": hackathons,
        "coding skills rating": coding_rating,
        "public speaking points": public_speaking,
        "self-learning capability?": self_learning,
        "Extra-courses did": extra_courses,
        "certifications": cert,
        "workshops": workshop,
        "reading and writing skills": read_write,
        "memory capability score": memory,
        "Interested subjects": sub,
        "interested career area": career_area,
        "Type of company want to settle in?": company_type,
        "Taken inputs from seniors or elders": senior_input,
        "Interested Type of Books": books,
        "Management or Technical": mgmt_tech,
        "hard/smart worker": hard_smart,
        "worked in teams ever?": teams,
        "Introvert": introvert
    }

    input_df = pd.DataFrame([user_inputs])

    try:
        prediction = pipeline.predict(input_df)[0]

        probs = pipeline.predict_proba(input_df)[0]
        classes = pipeline.classes_

        result_df = pd.DataFrame({
            "Career Role": classes,
            "Probability": probs * 100
        })

        result_df = result_df.sort_values(
            "Probability",
            ascending=False
        )

        st.success(f"🎯 Recommended Career: {prediction}")

        fig = px.bar(
            result_df.head(5),
            x="Probability",
            y="Career Role",
            orientation="h",
            text="Probability"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Top 3 Recommendations")

        for i in range(min(3, len(result_df))):
            st.write(
                f"{i+1}. {result_df.iloc[i]['Career Role']} "
                f"({result_df.iloc[i]['Probability']:.2f}%)"
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
