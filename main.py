import json
import os
from enum import Enum
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

load_dotenv()

# STEP 1: Model and environment setup
# - Load environment variables, initialize LLM (low temperature for consistent outputs)
# - Initialize embedding model used for RAG
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # low temperature for consistent outputs

# Small, inexpensive embedding model for RAG
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Check if the OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Add API Key into -> .env")

# Load Spacy NER model
print("Loading NER model...")
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading - spacy: python -m spacy download en_core_web_md")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")


# STEP 2: Data preparation and RAG indexing
def get_embedding(text: str) -> List[float]:
    """Create embeddings for the text"""
    text = text.replace("\n", " ")
    return embeddings_model.embed_query(text)


def prepare_trips_engine(trips_file: str) -> pd.DataFrame:
    """Loads trips, normalizes data and creates a vector index"""

    global df
    try:
        df = pd.read_json(trips_file)
    except ValueError:
        print('Error loading JSON file. Please check the file format.')

    df['trip_ref_id'] = df.index
    # Normalize city and country names to lowercase
    df['City_lower'] = df['City'].str.lower()
    df['Country_lower'] = df['Country'].str.lower()

    df['rag_content'] = df.apply(
        lambda x: f"Trip ID {x['trip_ref_id']}: Trip to {x['City']}, {x['Country']}. "
                  f"Activities: {', '.join(x['Extra activities'])}. "
                  f"Details: {x['Trip details']}", axis=1
    )

    # RAG indexing
    embeddings = []
    # We use tqdm for the progress bar
    for text in tqdm(df['rag_content'], desc="Creating vectors (Embeddings)"):
        embeddings.append(get_embedding(text))

    df['vector'] = list(embeddings)
    return df


# STEP 3: AI functions and output parsing

# Enum to ensure model outputs map to a strict set of sentiment labels
class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewAnalysis(BaseModel):
    sentiment: SentimentEnum = Field(
        description="The sentiment of the review. Must be explicitly 'positive', 'negative', or 'neutral'.")
    summary: str = Field(description="One sentence summary of the review")


parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

PROMPT_TEMPLATE = """
You are an expert Customer Experience AI. Analyze the following hotel review.
Strictly categorize sentiment as 'positive', 'negative', or 'neutral'. 
Do NOT use 'mixed'. If there are pros and cons, decide which prevails or use 'neutral'.

Review: "{review_text}"

{format_instructions}
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["review_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser


def extract_ner_locations(text: str) -> List[str]:
    """Extracts unique locations (GPE) using SpaCy."""
    doc = nlp(text)
    return list(set([ent.text.strip() for ent in doc.ents if ent.label_ == "GPE"]))


# STEP 4: Hybrid matching (NER + RAG)
def find_best_3_trips(review_text: str, ner_locs: List[str], trips_df: pd.DataFrame) -> List[Tuple[Any, str]]:
    """
    Hybrid algorithm within recommendation pipeline:
    Phase 1: NER hard filter (match by city or country)
    Phase 2: RAG semantic search (rank by embedding similarity)
    """
    recommendations = []
    seen_indices = set()

    # Phase 1: NER (Hard Filter)
    for loc in ner_locs:
        loc_lower = loc.lower()
        matches = trips_df[
            (trips_df['City_lower'] == loc_lower) |
            (trips_df['Country_lower'] == loc_lower)
            ]

        for idx, row in matches.iterrows():
            if idx not in seen_indices:
                recommendations.append((row, "NER_Location_Match"))
                seen_indices.add(idx)

    # If we already have 3 recommendations from NER, return them
    if len(recommendations) >= 3:
        return recommendations[:3]

    # Phase 2: RAG semantic search (fallback / complement to NER)
    # Create query embedding for the review
    review_vector = np.array(embeddings_model.embed_query(review_text)).reshape(1, -1)

    # Trips vectors matrix
    trips_matrix = np.array(trips_df['vector'].tolist())

    # Cosine similarity
    scores = cosine_similarity(review_vector, trips_matrix)[0]
    best_indices = np.argsort(scores)[::-1]

    for idx in best_indices:
        if len(recommendations) >= 3: break
        if idx not in seen_indices:
            recommendations.append((trips_df.iloc[idx], "RAG_Activity_Match"))
            seen_indices.add(idx)

    return recommendations[:3]


# ---  MAIN LOGIC (SINGLE REVIEW) ---

def analyze_single_review(text: str, score: int, review_id: str, trips_df: pd.DataFrame) -> Dict[str, Any]:
    """Analysis of a single review."""

    # 1. AI analysis
    try:
        llm_res = chain.invoke({"review_text": text})
        ner_locs = extract_ner_locations(text)

        sentiment_str = llm_res.sentiment.value
    except Exception as e:
        return {"error": str(e), "review_snippet": text[:50], "action": "ERROR"}

    # 2. Guardrails & logic adjustments
    final_sentiment = llm_res.sentiment.lower()
    if score <= 2:
        final_sentiment = "negative"
    elif score >= 4:
        final_sentiment = "positive"
    else:
        final_sentiment = sentiment_str

    # 3. Decision
    # action = "DISCOUNT" if final_sentiment in ["negative", "neutral"] else "RECOMMENDATION"
    if final_sentiment == "positive":
        action = "RECOMMENDATION"
    else:
        action = "DISCOUNT"

    result = {
        "review_id": review_id,
        "review_snippet": text[:50].replace("\n", " ") + "...",
        "score": score,
        "pred_sentiment": final_sentiment,
        "action": action,
        "summary": llm_res.summary,
        "ner_locations": ", ".join(ner_locs)
    }

    # 4. Execute action
    if action == "RECOMMENDATION":
        recs = find_best_3_trips(text, ner_locs, trips_df)
        for i, (trip, reason) in enumerate(recs):
            prefix = f"Rec_{i + 1}"
            result[f"{prefix}_Ref_ID"] = trip['trip_ref_id']
            result[f"{prefix}_City"] = trip['City']
            result[f"{prefix}_Country"] = trip['Country']
            result[f"{prefix}_Reason"] = reason
            result[f"{prefix}_Activities"] = ", ".join(trip['Extra activities'])
    else:
        result["Discount_Code"] = "SORRY_2025"

    return result


# --- EVALUATION AND VISUALIZATION ---

def visualize_results(csv_path):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if df.empty: return

    os.makedirs('output', exist_ok=True)
    sns.set_style("whitegrid")

    # Confusion matrix
    if 'true_sentiment' in df.columns:
        valid = df[df['true_sentiment'] != 'unknown']
        if not valid.empty:
            plt.figure(figsize=(6, 5))
            labels = sorted(valid['true_sentiment'].unique())
            cm = confusion_matrix(valid['true_sentiment'], valid['pred_sentiment'], labels=labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig('output/confusion_matrix.png')
            print("📈 Saved: output/confusion_matrix.png")

    # Sources of recommendations
    reasons = []
    for col in ['Rec_1_Reason', 'Rec_2_Reason', 'Rec_3_Reason']:
        if col in df.columns:
            reasons.extend(df[col].dropna().tolist())

    if reasons:
        plt.figure(figsize=(8, 5))
        sns.countplot(y=reasons, palette='viridis')
        plt.title('Sources of recommendations (NER vs RAG)')
        plt.tight_layout()
        plt.savefig('output/reasons_distribution.png')
        print("📈 Saved: output/reasons_distribution.png")


# --- BATCH PROCESS ---

def run_batch_pipeline(trips_df, limit=None):
    INPUT_FILE = 'data/customer_surveys_hotels_1k.json'
    OUTPUT_FILE = 'output/final_results.csv'

    print(f"\nStarting processing (Limit: {limit})...")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            surveys = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {INPUT_FILE}")
        return
    except json.JSONDecodeError:
        print(f"Error: File {INPUT_FILE} is not valid JSON.")
        return

    data = surveys[:limit] if limit else surveys
    results = []

    for item in tqdm(data):
        # Safely retrieve data (using .get to guard against missing keys)
        r_id = item.get('id', 'unknown_id')
        text = item.get('review', "")
        score = item.get('customer_satisfaction_score', 3)
        true_sent = item.get('survey_sentiment', 'unknown')

        # Analysis
        res = analyze_single_review(text, score, r_id, trips_df)

        # Add ground truth
        res['true_sentiment'] = true_sent
        if res['true_sentiment'] != 'unknown':
            res['sentiment_match'] = (res['true_sentiment'] == res['pred_sentiment'])
        results.append(res)

    df_out = pd.DataFrame(results)
    # When saving, enforce utf-8 encoding
    df_out.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"💾 Saved results: {OUTPUT_FILE}")

    # Quick metric
    if 'sentiment_match' in df_out.columns:
        acc = df_out['sentiment_match'].mean() * 100
        print(f"📊 Accuracy: {acc:.2f}%")

    visualize_results(OUTPUT_FILE)


def run_manual_demo(trips_df):
    print("\n💡 DEMO MODE (type 'exit' to quit)")
    while True:
        text = input("\n📝 Review: ")
        if text.strip().lower() == 'exit': break

        # --- VALIDATION LOOP START ---
        while True:
            user_input = input("⭐ Rating (1-5): ")

            # Check if input is a valid integer
            try:
                score = int(user_input)
                # Check if it is within the 1-5 range
                if 1 <= score <= 5:
                    break
                else:
                    print("   ⚠️ Please enter a number between 1 and 5.")
            except ValueError:
                print("   ⚠️ Invalid input. Please enter a number.")
        # --- VALIDATION LOOP END ---

        print("⏳ Analyzing...")

        # Remember to include the dummy ID "manual_test" here!
        res = analyze_single_review(text, score, "manual_test", trips_df)

        print(f"\n--- RESULT ({res['action']}) ---")
        print(f"Sentiment: {res['pred_sentiment']}")
        print(f"Summary: {res['summary']}")

        if res['action'] == "RECOMMENDATION":
            for i in range(1, 4):
                if f"Rec_{i}_City" in res:
                    print(
                        f"   {i}. {res[f'Rec_{i}_City']} ({res[f'Rec_{i}_Reason']}) -> {res[f'Rec_{i}_Activities']}")
        else:
            print(f"🎟️ Discount code: {res.get('Discount_Code')}")


if __name__ == "__main__":
    # One-time initialization
    trips_df = prepare_trips_engine('data/trips_data.json')

    # run_batch_pipeline(trips_df)  # ,  limit = 100)

    run_manual_demo(trips_df)
