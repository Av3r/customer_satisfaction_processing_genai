import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Import the refactored classes from main.py
from main import TripEngine, ReviewAnalyzer, PipelineManager

# Load env variables
load_dotenv()

# Set Page Config
st.set_page_config(
    page_title="AI Travel Assistant Dashboard",
    page_icon="✈️",
    layout="wide"
)


# --- CACHING THE PIPELINE ---
@st.cache_resource
def load_pipeline():
    """
    Initializes the refactored services once.
    """
    if not os.path.exists('data/trips_data.json'):
        st.error("⚠️ `data/trips_data.json` not found.")
        return None

    # 1. Initialize Models
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Initialize Engine and Analyzer
    engine = TripEngine(embed_model)
    engine.prepare_data('data/trips_data.json')

    analyzer = ReviewAnalyzer(llm)

    # 3. Return the Manager
    return PipelineManager(engine, analyzer)


pipeline = load_pipeline()

# --- UI LAYOUT ---
st.title("🤖 AI Travel Agent & Analytics Dashboard")
st.markdown("""
**System Architecture:**
1.  **ReviewAnalyzer:** Uses GPT-4o-mini for sentiment & summary.
2.  **TripEngine:** Hybrid NER (SpaCy) + Vector Search (Cosine Similarity).
3.  **PipelineManager:** Orchestrates the flow and applies score-based guardrails.
""")

tabs = st.tabs(["🧪 Interactive Demo", "📊 Batch Analytics & Stats", "ℹ️ Input Data Info", "📈 Output Data Stats"])

# --- TAB 1: INTERACTIVE DEMO ---
with tabs[0]:
    st.header("Test a Single Review")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_review = st.text_area("Enter a Customer Review:",
                                   "I loved the mountains in Italy, but the hotel was loud. I want to see snowy peaks in Switzerland next time.",
                                   height=150)
    with col2:
        user_score = st.slider("Customer Score (1-5)", 1, 5, 4)
        st.caption("Score Logic: ≤2 forces Negative, ≥4 forces Positive.")

    if st.button("Analyze Review", type="primary"):
        if pipeline:
            with st.spinner("Processing through PipelineManager..."):
                res = pipeline.process_single(user_review, user_score)

            if "error" in res:
                st.error(f"Error: {res['error']}")
            else:
                st.subheader(f"Action: {res['action']}")

                m1, m2, m3 = st.columns(3)
                # Updated keys to match PipelineManager.process_single output
                m1.metric("Predicted Sentiment", res['pred_sentiment'].upper())
                m2.metric("User Score", res['score'])
                m3.metric("Result Type", "RECOMMENDATION" if res['action'] == "RECOMMENDATION" else "DISCOUNT")

                st.info(f"**Summary:** {res['summary']}")
                st.write(f"**Detected Locations (NER):** {res['ner_locations'] if res['ner_locations'] else 'None'}")

                if res['action'] == "RECOMMENDATION":
                    st.subheader("🌟 Top Recommendations")
                    for i in range(1, 4):
                        city_key = f"Rec_{i}_City"
                        if city_key in res:
                            with st.expander(f"{i}. {res[city_key]}, {res[f'Rec_{i}_Country']}"):
                                st.write(f"**Match Type:** {res[f'Rec_{i}_Reason']}")
                                st.write(f"**Activities:** {res[f'Rec_{i}_Activities']}")
                else:
                    st.warning(f"🎟️ Retention Offer Issued: `{res.get('Discount_Code')}`")

# --- TAB 2: BATCH ANALYTICS ---
with tabs[1]:
    st.header("Batch Processing & Evaluation")
    reviews_file = 'data/customer_surveys_hotels_1k.json'

    if os.path.exists(reviews_file):
        limit = st.slider("Select batch size:", 10, 1000, 50)

        if st.button("Run Batch Evaluation"):
            if pipeline:
                with open(reviews_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)[:limit]

                results = []
                progress_bar = st.progress(0)

                for i, item in enumerate(data):
                    # Use PipelineManager for consistent logic
                    pred = pipeline.process_single(
                        item.get('review', ''),
                        item.get('customer_satisfaction_score', 3),
                        r_id=item.get('id', 'unknown')
                    )

                    # Align ground truth labels
                    true_sent = str(item.get('survey_sentiment', 'unknown')).lower().strip()

                    row = {
                        "id": pred['review_id'],
                        "score": pred['score'],
                        "true_sentiment": true_sent,
                        "pred_sentiment": pred['pred_sentiment'],
                        "action": pred['action']
                    }
                    results.append(row)
                    progress_bar.progress((i + 1) / limit)

                df_res = pd.DataFrame(results)
                st.dataframe(df_res)

                # --- METRICS ---
                valid_df = df_res[df_res['true_sentiment'].isin(['positive', 'negative', 'neutral'])]
                if not valid_df.empty:
                    st.subheader("📈 Performance Metrics")
                    y_true, y_pred = valid_df['true_sentiment'], valid_df['pred_sentiment']
                    labels = ['positive', 'negative', 'neutral']

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred) * 100:.1f}%")
                    c2.metric("Precision (Weighted)",
                              f"{precision_score(y_true, y_pred, average='weighted', zero_division=0):.2f}")
                    c3.metric("Recall (Weighted)",
                              f"{recall_score(y_true, y_pred, average='weighted', zero_division=0):.2f}")

                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(confusion_matrix(y_true, y_pred, labels=labels),
                                annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    st.pyplot(fig)

                # --- SAVE TO FILE ---
                default_path = 'output/final_results.csv'
                valid_df.to_csv(default_path, index=False, encoding='utf-8')
                print(f"Results saved to {default_path}")

# --- TAB 3: DATA INFO ---
with tabs[2]:
    st.header("📂 Input Data Analysis")

    if os.path.exists(reviews_file):
        with open(reviews_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        df_raw = pd.DataFrame(raw_data)

        # --- 1. HIGH-LEVEL KPIs ---
        st.subheader("📊 Dataset Overview")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        kpi1.metric("Total Reviews", len(df_raw))
        kpi2.metric("Avg. Satisfaction Score", f"{df_raw['customer_satisfaction_score'].mean():.2f} / 5")

        # Calculate Sentiment % (if available)
        if 'survey_sentiment' in df_raw.columns:
            pos_count = df_raw['survey_sentiment'].apply(
                lambda x: 1 if str(x).lower().strip() in ['positive', 'pos'] else 0).sum()
            kpi3.metric("Positive Sentiment %", f"{(pos_count / len(df_raw)) * 100:.1f}%")

        # Avg Word Count
        df_raw['word_count'] = df_raw['review'].apply(lambda x: len(str(x).split()))
        kpi4.metric("Avg. Word Count", f"{df_raw['word_count'].mean():.0f} words")

        st.markdown("---")

        # --- 2. DISTRIBUTIONS (Row 1) ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Customer Score Distribution")
            fig_hist, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='customer_satisfaction_score', data=df_raw, palette='viridis', ax=ax)
            ax.set_title("Count of Reviews by Score (1-5)")
            ax.set_ylabel("Number of Reviews")
            st.pyplot(fig_hist)

        with col2:
            st.subheader("Ground Truth Sentiment Split")
            if 'survey_sentiment' in df_raw.columns:
                # Normalize labels for plotting
                df_raw['clean_sentiment'] = df_raw['survey_sentiment'].apply(lambda x: str(x).lower().strip())
                sentiment_counts = df_raw['clean_sentiment'].value_counts()

                fig_pie, ax = plt.subplots(figsize=(6, 4))
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140,
                       colors=sns.color_palette('pastel'))
                ax.set_title("Sentiment Labels in Dataset")
                st.pyplot(fig_pie)
            else:
                st.info("No 'survey_sentiment' column found in JSON.")

        # --- 3. DATA QUALITY CHECK (Row 2) ---
        st.subheader("🔍 Data Quality: Score vs. Sentiment Consistency")
        st.markdown("Are high scores (5) actually labeled 'Positive'? This heatmap reveals dirty data.")

        if 'clean_sentiment' in df_raw.columns:
            # Create a cross-tabulation (Confusion Matrix of Ground Truths)
            consistency_matrix = pd.crosstab(df_raw['customer_satisfaction_score'], df_raw['clean_sentiment'])

            fig_heat, ax = plt.subplots(figsize=(8, 3))
            sns.heatmap(consistency_matrix, annot=True, fmt='d', cmap='Reds', ax=ax)
            ax.set_title("Heatmap: Score (Rows) vs. Sentiment Label (Cols)")
            st.pyplot(fig_heat)

        st.markdown("---")

        # --- 4. TEXT ANALYSIS (Row 3) ---
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Review Length Distribution")
            fig_len, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df_raw['word_count'], bins=20, kde=True, color='teal', ax=ax)
            ax.set_title("Distribution of Review Lengths (Word Count)")
            st.pyplot(fig_len)

        with col4:
            st.subheader("Top 15 Frequent Words")
            # Simple manual stopword removal to avoid importing NLTK just for this
            stopwords = set(
                ['the', 'and', 'to', 'was', 'a', 'of', 'in', 'for', 'i', 'it', 'we', 'is', 'very', 'with', 'my', 'that',
                 'but', 'on', 'at', 'had', 'room', 'hotel', 'this'])

            all_words = " ".join(df_raw['review'].astype(str)).lower().split()
            filtered_words = [w for w in all_words if w.isalpha() and w not in stopwords]

            from collections import Counter

            common_words = Counter(filtered_words).most_common(15)

            words_df = pd.DataFrame(common_words, columns=['Word', 'Count'])

            fig_bar, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Count', y='Word', data=words_df, palette='coolwarm', ax=ax)
            ax.set_title("Most Common Words (Excluding Stopwords)")
            st.pyplot(fig_bar)

    else:
        st.error("Data file not found. Please ensure `data/customer_surveys_hotels_1k.json` exists.")

# --- TAB 4: OUTPUT DATA STATS & CONSISTENCY ---
with tabs[3]:
    st.header("🔄 Dynamic Input vs. Output Consistency")

    input_file = 'data/customer_surveys_hotels_1k.json'
    output_file = 'output/final_results.csv'

    if os.path.exists(output_file) and os.path.exists(input_file):
        # 1. Load the results from the most recent Batch Run
        df_out = pd.read_csv(output_file)
        batch_size = len(df_out)

        # 2. Load the Input data and slice it to match the Batch size
        with open(input_file, 'r', encoding='utf-8') as f:
            full_input = json.load(f)
            # Only use the number of rows that were actually processed
            df_in_subset = pd.DataFrame(full_input[:batch_size])

        # Normalize sentiment strings
        df_in_subset['sentiment_clean'] = df_in_subset['survey_sentiment'].str.lower().str.strip()
        df_out['sentiment_clean'] = df_out['pred_sentiment'].str.lower().str.strip()

        st.info(f"💡 Comparison based on the last batch run of **{batch_size}** rows.")

        # --- 3. SIDE-BY-SIDE DISTRIBUTIONS ---
        st.subheader("📊 Sentiment Distribution Shift")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"**Original Labels (First {batch_size} Rows)**")
            fig_in, ax_in = plt.subplots(figsize=(6, 4))
            sns.countplot(x='sentiment_clean', data=df_in_subset, palette='Blues',
                          ax=ax_in, order=['positive', 'neutral', 'negative'])
            st.pyplot(fig_in)

        with c2:
            st.markdown(f"**AI Predicted Labels (Batch Output)**")
            fig_out, ax_out = plt.subplots(figsize=(6, 4))
            sns.countplot(x='sentiment_clean', data=df_out, palette='Oranges',
                          ax=ax_out, order=['positive', 'neutral', 'negative'])
            st.pyplot(fig_out)

        st.markdown("---")

        # --- 4. GUARDRAIL HEATMAP ---
        st.subheader("🔍 Logic Consistency Heatmap")
        # Ensure we are checking the scores against predictions for this specific batch
        consistency_matrix = pd.crosstab(df_out['score'], df_out['sentiment_clean'])

        fig_heat, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(consistency_matrix, annot=True, fmt='d', cmap='Greens', ax=ax)
        plt.xlabel("AI Predicted Sentiment")
        plt.ylabel("User Score (1-5)")
        st.pyplot(fig_heat)

        # --- 5. COMPLIANCE METRICS ---
        low_score_compliant = len(df_out[(df_out['score'] <= 2) & (df_out['sentiment_clean'] == 'negative')])
        high_score_compliant = len(df_out[(df_out['score'] >= 4) & (df_out['sentiment_clean'] == 'positive')])

        m1, m2, m3 = st.columns(3)
        m1.metric("Batch Size", batch_size)

        # Calculate percentages safely to avoid division by zero
        low_total = len(df_out[df_out['score'] <= 2])
        high_total = len(df_out[df_out['score'] >= 4])

        m2.metric("Low Score Logic (≤2)", f"{(low_score_compliant / low_total * 100):.1f}%" if low_total > 0 else "N/A")
        m3.metric("High Score Logic (≥4)",
                  f"{(high_score_compliant / high_total * 100):.1f}%" if high_total > 0 else "N/A")

    else:
        st.warning(
            "⚠️ No batch results found. Please go to 'Batch Analytics & Stats' and click 'Run Batch Evaluation' first.")
