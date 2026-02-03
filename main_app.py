import streamlit as st
import pandas as pd
import pickle
from h1b_bot import get_qa_chain


# Page Config
st.set_page_config(
    page_title="H-1B Visa Analysis System",
    layout="wide",
    page_icon="🇺🇸"
)

# Load ML Model & Encoders
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


# Load Dropdown Options
def load_options(path):
    with open(path, "r", encoding="utf-8") as f:
        return sorted([line.strip() for line in f if line.strip()])

job_titles = load_options("options/job_titles.txt")
employers = load_options("options/employer_names.txt")
soc_codes = load_options("options/soc_codes.txt")


# App Header
st.markdown(
    "<h1 style='text-align:center;'>H-1B Visa Analysis & Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:gray;'>Prediction • Insights • Conversational AI</p>",
    unsafe_allow_html=True
)

st.markdown("---")


# Tabs
tab1, tab2 = st.tabs(["Visa Prediction", "H-1B Visa Chatbot"])

def generate_prediction_explanation(input_data, pred_label, probability):
    chain = get_qa_chain()

    prompt = f"""
    Explain the H-1B visa prediction result in simple human language.

    Prediction Result: {pred_label}
    Confidence: {probability:.2f}

    Applicant Details:
    - Job Title: {input_data['JOB_TITLE']}
    - Employer: {input_data['EMPLOYER_NAME']}
    - SOC Code: {input_data['SOC_CODE']}
    - Full Time Position: {input_data['FULL_TIME_POSITION']}
    - Prevailing Wage: {input_data['PREVAILING_WAGE']}
    - City: {input_data['CITY']}
    - State: {input_data['STATE']}

    Explain:
    - Why this result may have occurred
    - What factors helped or hurt the approval
    - Write in a friendly, non-technical tone
    - Do NOT give legal advice
    """

    return chain.invoke(prompt)


# TAB 1: VISA PREDICTION
with tab1:

    st.subheader("📋 Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        JOB_TITLE = st.selectbox("Job Title", job_titles)
        EMPLOYER_NAME = st.selectbox("Employer Name", employers)
        SOC_CODE = st.selectbox("SOC Code", soc_codes)
        FULL_TIME_POSITION = st.radio(
            "Full Time Position",
            ["Y", "N"],
            horizontal=True
        )

    with col2:
        PREVAILING_WAGE = st.number_input(
            "Prevailing Wage (USD)",
            min_value=0.0,
            step=1000.0
        )
        CITY = st.text_input("City")
        STATE = st.text_input("State")

    predict_btn = st.button("Predict Visa Status")

    if predict_btn:

        input_data = {
            "JOB_TITLE": JOB_TITLE,
            "FULL_TIME_POSITION": FULL_TIME_POSITION,
            "PREVAILING_WAGE": PREVAILING_WAGE,
            "CITY": CITY,
            "STATE": STATE,
            "EMPLOYER_NAME": EMPLOYER_NAME,
            "SOC_CODE": SOC_CODE
        }

        df = pd.DataFrame([input_data])

        # Standardize columns
        df.columns = df.columns.str.upper()
        feature_columns = [c.upper() for c in feature_columns]

        # Encode categorical columns
        for col, encoder in label_encoders.items():
            col = col.upper()
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        # Handle missing columns
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]


        pred_encoded = model.predict(df)[0]
        probability = model.predict_proba(df).max()

        label_map = {1: "CERTIFIED", 0: "DENIED"}
        pred_label = label_map[pred_encoded]

        # GenAI Explanation
        with st.spinner("Generating explanation..."):
            explanation = generate_prediction_explanation(
                input_data,
                pred_label,
                probability
            )

        st.markdown("---")
        st.subheader("Prediction Result")

        if pred_label == "CERTIFIED":
            st.success(" **Visa Status: CERTIFIED**")
        else:
            st.error(" **Visa Status: DENIED**")

        st.metric("Prediction Confidence", f"{probability:.2f}")

        st.subheader("AI Interpretation (GenAI)")

        with st.expander("Read detailed explanation"):
            if explanation and explanation.strip():
                st.write(explanation)
            else:
                st.write(
                    "The AI could not generate a detailed explanation at this time. "
                    "Please review the prediction confidence and input details."
                )


        with st.expander("Interpretation"):
            st.write(
                "This prediction is generated using a machine learning model "
                "trained on historical H-1B visa application data. "
                "It is intended for educational and decision-support purposes only."
            )


# TAB 2: CHATBOT
with tab2:

    st.subheader("H-1B Visa Chatbot")
    st.caption("Ask about employers, occupations, approval trends, and visa insights")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.chat_input("Ask your question")

    if question:
        st.session_state.chat_history.append(("user", question))

        with st.spinner("Thinking..."):
            chain = get_qa_chain()
            answer = chain.invoke(question)
            if answer and answer.strip():
                st.session_state.chat_history.append(("bot", answer))
            else:
                st.session_state.chat_history.append(
                    ("bot", "I couldn't find a clear answer for that. Try rephrasing the question.")
                )

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f"""
                <div style="
                    background:#2E7D32;
                    color:#FFFFFF;
                    padding:12px;
                    border-radius:10px;
                    margin-bottom:8px;
                ">
                    {msg}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="
                    background:#1E1E1E;
                    color:#FFFFFF;
                    padding:12px;
                    border-radius:10px;
                    border-left:4px solid #4CAF50;
                    margin-bottom:12px;
                ">
                    {msg}
                </div>
                """,
                unsafe_allow_html=True
            )


# Footer
st.markdown("---")
st.markdown(
    "<center style='color:gray;'>Educational tool • Not legal advice</center>",
    unsafe_allow_html=True
)
