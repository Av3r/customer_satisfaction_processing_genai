import json
import os
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import spacy
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# --- Models & Schemas ---

class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewAnalysis(BaseModel):
    sentiment: SentimentEnum = Field(description="The sentiment of the review.")
    summary: str = Field(description="One sentence summary of the review")


# --- Core Logic Classes ---

class TripEngine:
    """Handles data loading, embedding generation, and vector search."""

    @staticmethod
    def _load_spacy():
        try:
            return spacy.load("en_core_web_md")
        except OSError:
            os.system("python -m spacy download en_core_web_md")
            return spacy.load("en_core_web_md")

    def __init__(self, embeddings_model: OpenAIEmbeddings):
        self.embeddings_model = embeddings_model
        self.df = pd.DataFrame()
        self.nlp = self._load_spacy()

    def prepare_data(self, trips_file: str):
        try:
            self.df = pd.read_json(trips_file, encoding='utf-8')
            self.df['trip_ref_id'] = self.df.index
            self.df['City_lower'] = self.df['City'].str.lower()
            self.df['Country_lower'] = self.df['Country'].str.lower()

            self.df['rag_content'] = self.df.apply(
                lambda x: f"Trip ID {x['trip_ref_id']}: {x['City']}, {x['Country']}. "
                          f"Activities: {', '.join(x['Extra activities'])}. "
                          f"Details: {x['Trip details']}", axis=1
            )

            embeddings = [
                self.embeddings_model.embed_query(text.replace("\n", " "))
                for text in tqdm(self.df['rag_content'], desc="Creating vectors")
            ]
            self.df['vector'] = list(embeddings)
        except Exception as e:
            print(f"Error preparing TripEngine: {e}")

    def extract_ner_locations(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return list(set([ent.text.strip() for ent in doc.ents if ent.label_ == "GPE"]))

    def find_recommendations(self, review_text: str, ner_locs: List[str], limit: int = 3) -> List[Tuple[Any, str]]:
        """
            Hybrid algorithm within recommendation pipeline:
            Phase 1: NER hard filter (match by city or country)
            Phase 2: RAG semantic search (rank by embedding similarity)
        """
        recommendations = []
        seen_indices = set()

        # Phase 1: NER hard filter
        for loc in ner_locs:
            loc_l = loc.lower()
            matches = self.df[(self.df['City_lower'] == loc_l) | (self.df['Country_lower'] == loc_l)]
            for idx, row in matches.iterrows():
                if idx not in seen_indices and len(recommendations) < limit:
                    recommendations.append((row, "NER_Location_Match"))
                    seen_indices.add(idx)

        # Phase 2: RAG semantic search
        if len(recommendations) < limit:
            query_vec = np.array(self.embeddings_model.embed_query(review_text)).reshape(1, -1)
            trips_matrix = np.array(self.df['vector'].tolist())
            scores = cosine_similarity(query_vec, trips_matrix)[0]

            for idx in np.argsort(scores)[::-1]:
                if len(recommendations) >= limit: break
                if idx not in seen_indices:
                    recommendations.append((self.df.iloc[idx], "RAG_Activity_Match"))
                    seen_indices.add(idx)

        return recommendations


class ReviewAnalyzer:
    """Orchestrates LLM analysis and business logic guardrails."""

    def __init__(self, llm: ChatOpenAI):
        self.parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
        self.prompt = PromptTemplate(
            template="""
            You are an expert Customer Experience AI. Analyze the following hotel review.
            Strictly categorize sentiment as 'positive', 'negative', or 'neutral'. 
            Do NOT use 'mixed'. If there are pros and cons, decide which prevails or use 'neutral'.

            Review: "{review_text}"

            {format_instructions}
            """,
            input_variables=["review_text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt | llm | self.parser

    def analyze(self, text: str, score: int) -> Dict[str, Any]:
        try:
            analysis = self.chain.invoke({"review_text": text})
            # Business Guardrails
            final_sentiment = analysis.sentiment.value
            if score <= 2:
                final_sentiment = "negative"
            elif score >= 4:
                final_sentiment = "positive"

            return {
                "sentiment": final_sentiment,
                "summary": analysis.summary,
                "action": "RECOMMENDATION" if final_sentiment == "positive" else "DISCOUNT"
            }
        except Exception as e:
            return {"error": str(e), "sentiment": "neutral", "action": "DISCOUNT", "summary": "N/A"}


class PipelineManager:
    """High-level coordinator for batch and manual runs."""

    def __init__(self, engine: TripEngine, analyzer: ReviewAnalyzer):
        self.engine = engine
        self.analyzer = analyzer

    def process_single(self, text: str, score: int, r_id: str = "manual") -> Dict[str, Any]:
        analysis = self.analyzer.analyze(text, score)
        ner_locs = self.engine.extract_ner_locations(text)

        result = {
            "review_id": r_id,
            "review_snippet": f"{text[:50]}...",
            "score": score,
            "pred_sentiment": analysis["sentiment"],
            "action": analysis["action"],
            "summary": analysis["summary"],
            "ner_locations": ", ".join(ner_locs)
        }

        if analysis["action"] == "RECOMMENDATION":
            recs = self.engine.find_recommendations(text, ner_locs)
            for i, (trip, reason) in enumerate(recs):
                key = f"Rec_{i + 1}"
                result.update({
                    f"{key}_City": trip['City'],
                    f"{key}_Country": trip['Country'],
                    f"{key}_Reason": reason,
                    f"{key}_Activities": ", ".join(trip['Extra activities'])
                })
        else:
            result["Discount_Code"] = "SORRY_2026"

        return result

    def run_batch(self, input_path: str, output_path: str, limit: Optional[int] = None):
        with open(input_path, 'r', encoding='utf-8') as f:
            surveys = json.load(f)[:limit] if limit else json.load(f)

        results = []
        for item in tqdm(surveys, desc="Batch processing"):
            res = self.process_single(
                item.get('review', ""),
                item.get('customer_satisfaction_score', 3),
                item.get('id', 'unknown')
            )
            res['true_sentiment'] = item.get('survey_sentiment', 'unknown')
            results.append(res)

        df_out = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Results saved to {output_path}")


# --- Main Entry Point ---

def get_valid_score() -> int:
    """Ensures user input is an integer between 1 and 5."""
    while True:
        try:
            score = int(input("⭐ Rating (1-5): "))
            if 1 <= score <= 5:
                return score
            print("   ⚠️ Please enter a number between 1 and 5.")
        except ValueError:
            print("   ⚠️ Invalid input. Please enter a numeric value.")


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Missing OPENAI_API_KEY in .env")

    # 1. Setup Dependencies
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Initialize Engine and Analyzer
    engine = TripEngine(embed_model)
    engine.prepare_data('data/trips_data.json')

    analyzer = ReviewAnalyzer(llm)
    manager = PipelineManager(engine, analyzer)

    # 3. Execution Selection
    print("\n--- Travel AI Pipeline ---")
    print("1. Run Batch Processing")
    print("2. Run Manual Demo")
    choice = input("Select mode (1/2): ")

    if choice == "1":
        # BATCH RUN
        INPUT_FILE = 'data/customer_surveys_hotels_1k.json'
        OUTPUT_FILE = 'output/final_results.csv'
        manager.run_batch(INPUT_FILE, OUTPUT_FILE, limit=50)  # Set limit=None for full run

    elif choice == "2":
        # MANUAL DEMO
        print("\n💡 DEMO MODE (type 'exit' to quit)")
        while True:
            text = input("\n📝 Review: ")
            if text.strip().lower() == 'exit':
                break

            score = get_valid_score()
            print("⏳ Analyzing...")

            result = manager.process_single(text, score, r_id="manual_test")

            print(f"\n--- RESULT ({result['action']}) ---")
            print(f"Sentiment: {result['pred_sentiment']}")
            print(f"Summary:   {result['summary']}")

            if result['action'] == "RECOMMENDATION":
                for i in range(1, 4):
                    city_key = f"Rec_{i}_City"
                    if city_key in result:
                        print(f"   {i}. {result[city_key]} ({result[f'Rec_{i}_Reason']})")
            else:
                print(f"🎟️ Discount code: {result.get('Discount_Code')}")
    else:
        print("Invalid selection. Exiting.")


if __name__ == "__main__":
    main()
