import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

# --------------------------
# Load Models
# --------------------------
load_dotenv()

# Clause Classifier
clause_model = joblib.load("legal_clause_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Contract Search Data
contracts_df = pd.read_pickle("contracts_metadata.pkl")

# Embedding Model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Groq Client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# --------------------------
# PAGE CONFIG
# --------------------------

st.set_page_config(
    page_title="Legal Contract Intelligence Platform",
    page_icon="⚖️",
    layout="wide"
)

# --------------------------
# CUSTOM CSS
# --------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.big-title {
    font-size:40px;
    font-weight:bold;
    color:#1f4e79;
}

.metric-card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}

.stButton>button {
    background:#1f4e79;
    color:white;
    border-radius:10px;
    height:50px;
    width:100%;
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# HEADER
# --------------------------

st.markdown(
    '<p class="big-title">⚖️ AI Legal Contract Intelligence Platform</p>',
    unsafe_allow_html=True
)

st.caption(
    "Clause Classification • Risk Scoring • Similarity Search • GenAI Legal Assistant"
)

st.divider()

# --------------------------
# SIDEBAR
# --------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Clause Classifier",
        "Risk Analysis",
        "Contract Search",
        "Legal AI Chatbot"
    ]
)

# ===================================================
# DASHBOARD
# ===================================================

if page == "Dashboard":

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric(
            "Contracts",
            "509"
        )

    with col2:
        st.metric(
            "Clauses",
            "5,694"
        )

    with col3:
        st.metric(
            "Risk Rules",
            "8"
        )

    with col4:
        st.metric(
            "Vector Chunks",
            "39,020"
        )

    st.divider()

    st.subheader("Platform Overview")

    st.info("""
    This platform automatically:

    ✔ Classifies legal clauses

    ✔ Calculates contract risk score

    ✔ Finds similar contracts

    ✔ Answers legal questions using RAG + Gemini
    """)

# ===================================================
# CLAUSE CLASSIFIER
# ===================================================

elif page == "Clause Classifier":

    st.subheader("📑 Clause Classification")

    clause_text = st.text_area(
        "Enter Clause Text"
    )

    if st.button("Classify Clause"):

        if clause_text.strip():

            x = tfidf.transform([clause_text])

            prediction = clause_model.predict(x)[0]

            probs = clause_model.predict_proba(x)[0]

            confidence = np.max(probs) * 100

            st.success(
                f"Predicted Clause: {prediction}"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

# ===================================================
# RISK ANALYSIS
# ===================================================

elif page == "Risk Analysis":

    st.subheader("⚠ Risk Analysis")

    contract_text = st.text_area(
        "Paste Contract Text"
    )

    if st.button("Analyze Risk"):

        score = 0
        factors = []

        text = contract_text.lower()

        if "non-compete" in text:
            score += 20
            factors.append("Non-Compete")

        if "change of control" in text:
            score += 15
            factors.append("Change Of Control")

        if "exclusive" in text:
            score += 15
            factors.append("Exclusivity")

        if "assign" in text:
            score += 5
            factors.append("Anti-Assignment")

        st.metric(
            "Risk Score",
            score
        )

        if score >= 50:
            st.error("High Risk")
        elif score >= 20:
            st.warning("Medium Risk")
        else:
            st.success("Low Risk")

        st.subheader(
            "Detected Risk Factors"
        )

        st.write(factors)
# ===================================================
# CONTRACT SEARCH
# ===================================================

elif page == "Contract Search":

    st.subheader(
        "🔍 Similar Contract Search"
    )

    query = st.text_area(
        "Paste Contract Text"
    )

    if st.button(
        "Find Similar Contracts"
    ):

        contract_embeddings = embedding_model.encode(
            contracts_df["clean_text"].tolist(),
            show_progress_bar=False
        )

        query_embedding = embedding_model.encode(
            [query]
        )

        similarities = cosine_similarity(
            query_embedding,
            contract_embeddings
        )[0]

        top_idx = similarities.argsort()[-5:][::-1]

        results = pd.DataFrame({
            "contract_title":
            contracts_df.iloc[top_idx]["contract_title"].values,

            "similarity_percent":
            np.round(
                similarities[top_idx] * 100,
                2
            )
        })

        st.dataframe(
            results,
            use_container_width=True
        )

# ===================================================
# LEGAL AI CHATBOT
# ===================================================

elif page == "Legal AI Chatbot":

    st.subheader("🤖 Legal AI Assistant")

    question = st.text_input(
        "Ask a legal question"
    )

    if st.button("Generate Answer"):

        with st.spinner("Generating Answer..."):

            response = client.chat.completions.create(

                model="llama-3.3-70b-versatile",

                messages=[
                    {
                        "role":"user",
                        "content":question
                    }
                ]
            )

            st.write(
                response.choices[0].message.content
            )