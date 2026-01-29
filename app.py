import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from dotenv import load_dotenv

# --- IMPORT FROM YOUR MAIN.PY ---
# This prevents code duplication. If you fix logic in main.py, it updates here too.
from main import TravelAssistant, SentimentEnum

# Load env variables
load_dotenv()

# Set Page Config
st.set_page_config(
    page_title="AI Travel Assistant Dashboard",
    page_icon="✈️",
    layout="wide"
)


# --- CACHING THE MODEL ---
@st.cache_resource
def load_assistant():
    """
    Initializes the model only once to save resources.
    """
    if not os.path.exists('data/trips_data.json'):
        st.error("⚠️ `data/trips_data.json` not found. Please upload or place the file.")
        return None

    # We use the class directly from main.py
    return TravelAssistant(trips_file='data/trips_data.json')


assistant = load_assistant()

# --- UI LAYOUT ---

st.title("🤖 AI Travel Agent & Analytics Dashboard")
st.markdown("""
**How it works:**
1.  **Intent Extraction:** Uses **GPT-4o-mini** (via `main.py`) to read reviews.
2.  **Hybrid Filtering:** Combines Semantic Search (**FAISS**) with Rules.
3.  **Logic Gate:** Overrides sentiment based on score (e.g., Score 1 is always Negative).
""")

tabs = st.tabs(["🧪 Interactive Demo", "📊 Batch Analytics & Stats", "ℹ️ Input Data Info"])

# --- TAB 1: INTERACTIVE DEMO ---
with tabs[0]:
    st.header("Test a Single Review")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_review = st.text_area("Enter a Customer Review:",
                                   "I really hated my trip to Paris, it was too crowded. I want to go somewhere quiet like a beach in Thailand.",
                                   height=150)
    with col2:
        user_score = st.slider("Customer Score (1-5)", 1, 5, 2)
        st.caption("Scores ≤ 2 force 'Negative', Scores ≥ 5 force 'Positive'.")

    if st.button("Analyze Review", type="primary"):
        if assistant:
            with st.spinner("Analyzing intent and searching vector database..."):
                # CALLING THE BACKEND LOGIC
                res = assistant.analyze_review(user_review, user_score)

            if "error" in res:
                st.error(f"Error: {res['error']}")
            else:
                # Display Results
                st.subheader(f"Action: {res['action']}")

                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Sentiment", res['sentiment_detected'].upper())  # Updated key to match main.py
                m2.metric("Customer Score", res['score'])
                m3.metric("Discount Code", res.get("Discount_Code", "N/A"))

                # Intent details
                st.info(f"**Summary:** {res['summary']}")
                st.write(f"**😊 Wants:** {res['wants']}")
                st.write(f"**😡 Avoids:** {res['avoids']}")

                # Recommendations
                if res['action'] == "RECOMMENDATION":
                    st.subheader("🌟 Recommendations")
                    # We iterate 1-3 because main.py flattens them.
                    # Let's reconstruct them slightly for display or just access the keys.
                    for i in range(1, 4):
                        city_key = f"Rec_{i}_City"
                        if city_key in res:
                            with st.expander(f"{i}. {res[city_key]}, {res[f'Rec_{i}_Country']}"):
                                st.write(f"**Reason:** {res[f'Rec_{i}_Reason']}")
                                # Note: 'Activities' and 'Details' might need to be added to main.py's return
                                # if you want them visible here, or we accept the summary data.

# --- TAB 2: BATCH ANALYTICS ---
with tabs[1]:
    st.header("Batch Processing & Evaluation")

    reviews_file = 'data/customer_surveys_hotels_1k.json'

    if os.path.exists(reviews_file):
        st.success(f"Found dataset: `{reviews_file}`")
        limit = st.slider("Number of reviews to process:", 10, 200, 50)

        if st.button("Run Batch Evaluation"):
            if assistant:
                with open(reviews_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)[:limit]

                results = []
                progress_bar = st.progress(0)

                for i, item in enumerate(data):
                    # Prepare inputs
                    r_id = item.get('id')
                    txt = item.get('review', '')
                    scr = item.get('customer_satisfaction_score', 3)

                    # --- FIX: ROBUST NORMALIZATION ---
                    # 1. Get raw value, force string, lowercase, and strip spaces
                    raw_sent = str(item.get('survey_sentiment', 'unknown')).lower().strip()

                    # 2. Map variations if necessary (optional but safe)
                    if raw_sent in ['positive', 'pos']:
                        true_sent = 'positive'
                    elif raw_sent in ['negative', 'neg']:
                        true_sent = 'negative'
                    elif raw_sent in ['neutral', 'neu']:
                        true_sent = 'neutral'
                    else:
                        true_sent = 'unknown'

                    # RUN LOGIC
                    pred = assistant.analyze_review(txt, scr)

                    # Store Data
                    row = {
                        "id": r_id,
                        "score": scr,
                        "true_sentiment": true_sent,
                        "pred_sentiment": str(pred['sentiment_detected']).lower(),  # Force lower
                        "action": pred['action'],
                        "match_reason": pred.get('Rec_1_Reason', "N/A")
                    }
                    results.append(row)
                    progress_bar.progress((i + 1) / limit)

                df_res = pd.DataFrame(results)
                st.dataframe(df_res.head())

                # --- STATISTICS ---
                st.markdown("---")
                st.subheader("📈 Data Science Metrics")

                # Filter strictly for valid labels
                valid_df = df_res[df_res['true_sentiment'].isin(['positive', 'negative', 'neutral'])]

                if not valid_df.empty:
                    y_true = valid_df['true_sentiment']
                    y_pred = valid_df['pred_sentiment']
                    labels = ['positive', 'negative', 'neutral']

                    # Metrics
                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accuracy", f"{acc * 100:.1f}%")
                    c2.metric("Precision", f"{prec:.2f}")
                    c3.metric("Recall", f"{rec:.2f}")

                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred, labels=labels)

                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=labels, yticklabels=labels, ax=ax)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    st.pyplot(fig)
                else:
                    st.error("⚠️ All ground truth labels are 'unknown'. Please check your JSON file content.")
                    st.write("First 5 raw sentiments found in file:",
                             [str(x.get('survey_sentiment')) for x in data[:5]])

# --- TAB 3: INPUT DATA INFO ---
with tabs[2]:
    st.header("📂 Input Data Analysis")
    if os.path.exists(reviews_file):
        with open(reviews_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        df_raw = pd.DataFrame(raw_data)

        c1, c2 = st.columns(2)
        c1.metric("Total Reviews", len(df_raw))
        c2.metric("Avg Score", f"{df_raw['customer_satisfaction_score'].mean():.2f}")

        st.subheader("Score Distribution")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
        sns.histplot(df_raw['customer_satisfaction_score'], bins=5, kde=False, color='teal', ax=ax_hist)
        st.pyplot(fig_hist)
