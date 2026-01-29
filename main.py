import json
import logging
import os
from enum import Enum
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from tqdm import tqdm  # Progress bar for batch processing

# --- CONFIGURATION & SETUP ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in your .env file")


# --- DATA MODELS ---

class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class CustomerIntent(BaseModel):
    """
    Structured output to capture not just sentiment, but actionable intent.
    Distinguishes between what users want vs. what they want to avoid.
    """
    sentiment: SentimentEnum = Field(
        description="The overall sentiment of the review."
    )
    summary: str = Field(
        description="A concise one-sentence summary of the review."
    )
    desired_destinations: List[str] = Field(
        description="Specific cities or countries the user explicitly wants to visit next.",
        default_factory=list
    )
    disliked_destinations: List[str] = Field(
        description="Cities or countries the user disliked, visited recently, or wants to avoid.",
        default_factory=list
    )
    activity_keywords: List[str] = Field(
        description="Key activities mentioned as positive preferences (e.g. 'hiking', 'museums', 'beach').",
        default_factory=list
    )


# --- CORE LOGIC CLASS ---

class TravelAssistant:
    def __init__(self, trips_file: str, model_name: str = "gpt-4o-mini"):
        """
        Initializes the AI Travel Assistant.
        Loads data, prepares the vector store, and sets up the LLM chain.
        """
        self.trips_df = self._load_data(trips_file)

        # Initialize AI Components
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Initialize Logic components
        self.parser = PydanticOutputParser(pydantic_object=CustomerIntent)
        self.chain = self._build_analysis_chain()
        self.vector_store = self._build_vector_store()

    def _load_data(self, filepath: str) -> pd.DataFrame:
        """Loads and normalizes the trips dataset."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Trips file not found: {filepath}")

        try:
            df = pd.read_json(filepath)
            df['trip_ref_id'] = df.index
            # Normalize for matching
            df['City_norm'] = df['City'].str.lower().str.strip()
            df['Country_norm'] = df['Country'].str.lower().str.strip()
            logger.info(f"Loaded {len(df)} trips from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _build_vector_store(self) -> FAISS:
        """
        Builds a FAISS index for fast semantic retrieval.
        Runs purely on CPU (requires `pip install faiss-cpu`).
        """
        logger.info("Building Vector Index (CPU)...")

        # Construct rich semantic text for embedding
        texts = [
            f"Trip to {row['City']}, {row['Country']}. "
            f"Activities: {', '.join(row['Extra activities'])}. "
            f"Details: {row['Trip details']}"
            for _, row in self.trips_df.iterrows()
        ]

        # Store metadata to retrieve full trip details later
        metadatas = self.trips_df.to_dict('records')

        # Create Index
        return FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)

    def _build_analysis_chain(self):
        """Creates the LangChain for intent extraction."""
        template = """
        You are an expert Customer Experience AI. Analyze the following hotel review.

        Your Goal:
        1. Determine the sentiment (positive, negative, neutral).
        2. Extract distinct LOCATIONS. Crucially, distinguish between where the user WANTS to go 
           vs. where they disliked or just returned from.
        3. Extract activity preferences (keywords).

        Review: "{review_text}"

        {format_instructions}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["review_text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        return prompt | self.llm | self.parser

    def analyze_review(self, review_text: str, score: int, review_id: str = "unknown") -> Dict[str, Any]:
        """
        Main entry point for analyzing a single review.
        Orchestrates Analysis -> Logic Gate -> Recommendation/Discount.
        """
        # 1. AI Analysis (Intent Extraction)
        try:
            analysis: CustomerIntent = self.chain.invoke({"review_text": review_text})
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {"error": str(e), "action": "ERROR"}

        # 2. Logic Gate (Hard rules override LLM sentiment)
        # If score is very low/high, we force the sentiment category
        final_sentiment = analysis.sentiment
        if score <= 2:
            final_sentiment = SentimentEnum.NEGATIVE
        elif score >= 5:
            final_sentiment = SentimentEnum.POSITIVE

        # 3. Decision
        action = "RECOMMENDATION" if final_sentiment == SentimentEnum.POSITIVE else "DISCOUNT"

        result = {
            "review_id": review_id,
            "snippet": review_text[:50].replace("\n", " ") + "...",
            "score": score,
            "sentiment_detected": final_sentiment,
            "action": action,
            "summary": analysis.summary,
            "wants": ", ".join(analysis.desired_destinations),
            "avoids": ", ".join(analysis.disliked_destinations),
            "activities": ", ".join(analysis.activity_keywords)
        }

        # 4. Execute Action
        if action == "RECOMMENDATION":
            recs = self._get_recommendations(analysis)
            # Flatten recommendations for easier CSV/Display usage
            for i, rec in enumerate(recs):
                prefix = f"Rec_{i + 1}"
                result[f"{prefix}_City"] = rec['City']
                result[f"{prefix}_Country"] = rec['Country']
                result[f"{prefix}_Reason"] = rec['Match_Reason']
        else:
            result["Discount_Code"] = "SORRY_2025"

        return result

    def _get_recommendations(self, intent: CustomerIntent, k: int = 3) -> List[Dict]:
        """
        Smart Recommendation Engine.
        1. Construct a semantic query based on desires + activities.
        2. Retrieve more candidates than needed (k*2).
        3. Filter OUT disliked locations.
        """

        # Construct query: prioritize explicit desires, then activities
        query_parts = intent.desired_destinations + intent.activity_keywords
        query_text = " ".join(query_parts)

        if not query_text.strip():
            # Fallback if review was vague but positive
            query_text = "Highly rated amazing travel experiences"

        # Search Vector DB
        # We fetch 2x the needed amount to allow for filtering
        docs_and_scores = self.vector_store.similarity_search_with_score(query_text, k=k * 2)

        recommendations = []
        avoid_set = {loc.lower().strip() for loc in intent.disliked_destinations}
        seen_ids = set()

        for doc, score in docs_and_scores:
            trip = doc.metadata
            trip_id = trip['trip_ref_id']
            city_norm = trip.get('City_norm', '')
            country_norm = trip.get('Country_norm', '')

            # FILTER: Skip duplicates
            if trip_id in seen_ids:
                continue

            # FILTER: Skip explicitly disliked locations
            if city_norm in avoid_set or country_norm in avoid_set:
                continue

            # Determine reason
            reason = "Semantic Match"
            # Simple check if it matches explicit desire (could be improved with fuzzy matching)
            for desire in intent.desired_destinations:
                if desire.lower() in city_norm or desire.lower() in country_norm:
                    reason = "Explicit Destination Match"
                    break

            recommendations.append({
                **trip,
                "Match_Reason": reason,
                "Score": float(score)
            })

            seen_ids.add(trip_id)
            if len(recommendations) >= k:
                break

        return recommendations


# --- BATCH PROCESSOR ---

def run_batch_pipeline(assistant: TravelAssistant, limit: Optional[int] = None):
    """
    Processes a JSON file of reviews in the background and saves to CSV.
    No interactive chat.
    """
    INPUT_FILE = 'data/customer_surveys_hotels_1k.json'
    OUTPUT_FILE = 'output/final_results.csv'

    os.makedirs('output', exist_ok=True)
    print(f"\n🚀 Starting BATCH processing (Limit: {limit})...")
    print(f"📂 Reading from: {INPUT_FILE}")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            surveys = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {INPUT_FILE}")
        return

    data = surveys[:limit] if limit else surveys
    results = []

    # Use tqdm for a progress bar
    for item in tqdm(data, desc="Processing Reviews"):
        # Safely retrieve data
        r_id = item.get('id', 'unknown_id')
        text = item.get('review', "")
        score = item.get('customer_satisfaction_score', 3)

        # Analyze
        res = assistant.analyze_review(text, score, r_id)

        # Append ground truth for comparison if available
        res['true_sentiment'] = item.get('survey_sentiment', 'unknown')
        results.append(res)

    # Save to CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\n✅ Done! Saved results to: {OUTPUT_FILE}")


# --- DEMO RUNNER ---

def run_manual_demo(assistant: TravelAssistant):
    print("\n💡 DEMO MODE (type 'exit' to quit)")
    print("   Note: Try reviews like 'I hated Paris, I want a beach vacation.'")

    while True:
        text = input("\n📝 Review: ")
        if text.strip().lower() == 'exit':
            break

        while True:
            try:
                score_input = input("⭐ Rating (1-5): ")
                score = int(score_input)
                if 1 <= score <= 5: break
                print("   ⚠️ Enter number 1-5.")
            except ValueError:
                print("   ⚠️ Invalid input.")

        print("⏳ Analyzing...")
        res = assistant.analyze_review(text, score, "manual_test")

        print(f"\n--- RESULT: {res['action']} ---")
        print(f"Sentiment: {res['sentiment_detected']}")
        print(f"Summary:   {res['summary']}")

        if res['action'] == "RECOMMENDATION":
            print(f"Intent detected: Wants [{res['wants']}], Avoids [{res['avoids']}]")
            print("\nRecommended Trips:")
            for i in range(1, 4):
                key_city = f"Rec_{i}_City"
                if key_city in res:
                    print(f"   {i}. {res[key_city]}, {res[f'Rec_{i}_Country']}")
                    print(f"      Reason: {res[f'Rec_{i}_Reason']}")
        else:
            print(f"🎟️ Discount Code: {res.get('Discount_Code')}")


if __name__ == "__main__":
    # Ensure data directory exists
    if not os.path.exists('data/trips_data.json'):
        print("❌ Error: 'data/trips_data.json' not found. Please ensure the data file exists.")
    else:
        # Initialize the Assistant (Load Logic & Data once)
        try:
            assistant = TravelAssistant(trips_file='data/trips_data.json')

            # --- CHOOSE MODE HERE ---
            # Uncomment the line below to run the Batch Process
            run_batch_pipeline(assistant, limit=20)  # Set limit=10 for testing

            # Comment the line below to disable the Manual Demo
            # run_manual_demo(assistant)

        except Exception as e:
            print(f"❌ Initialization Failed: {e}")
