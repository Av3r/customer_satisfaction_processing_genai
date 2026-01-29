# AI Travel Assistant & Analytics Dashboard

This project is an advanced AI-driven customer experience platform that analyzes hotel reviews to determine sentiment and provide personalized travel recommendations or retention offers. It utilizes a hybrid approach combining **Natural Language Processing (NLP)**, **Large Language Models (LLMs)**, and **Retrieval-Augmented Generation (RAG)**.

## 🚀 System Architecture

The application is built on three core pillars:

1. **ReviewAnalyzer**: Utilizes `gpt-4o-mini` to categorize sentiment (Positive, Negative, Neutral) and generate concise summaries of customer feedback.
2. **TripEngine**: A hybrid engine that uses **SpaCy** for Named Entity Recognition (NER) to find specific locations and **Vector Search (Cosine Similarity)** to match review content with available trips.
3. **PipelineManager**: Orchestrates the workflow, applying business guardrails (e.g., forcing negative sentiment for low satisfaction scores) and determining whether to issue a recommendation or a discount code.

## 🛠️ Key Features

* **Interactive Demo**: Test individual reviews in real-time to see how the AI classifies sentiment and suggests destinations.
* **Batch Processing**: Analyze large datasets (JSON) to evaluate model performance across hundreds of reviews simultaneously.
* **Analytics Dashboard**: Visualize data quality, sentiment distribution, and performance metrics like Accuracy, Precision, and Recall using `matplotlib` and `seaborn`.
* **Hybrid Recommendation**: Combines hard-filter location matching with semantic activity matching.

## 📦 Project Structure

* `app.py`: The Streamlit-based web interface and dashboard.
* `main.py`: The core engine containing the OOP logic, AI chains, and data processing services.
* `data/`: Directory containing `trips_data.json` (trip database) and `customer_surveys_hotels_1k.json` (source reviews).
* `output/`: Stores generated CSV reports from batch runs.

## 🔧 Installation & Setup

1. **Environment Variables**: Create a `.env` file in the root directory and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here

```


2. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


3. **Download NLP Model**:
```bash
python -m spacy download en_core_web_md

```



## 📈 Usage

### Running the Web Dashboard

To launch the interactive Streamlit interface:

```bash
streamlit run app.py

```

### Running the CLI Pipeline

To run batch processing or a manual terminal-based demo:

```bash
python main.py

```

## 🧪 Evaluation Metrics

The system provides built-in evaluation tools in the **Batch Analytics** tab of the dashboard, including:

* **Confusion Matrix**: Visualizes the alignment between ground truth sentiment and AI predictions.
* **Precision/Recall**: Provides weighted metrics to assess the reliability of classification.
* **Data Quality Heatmap**: Compares numerical satisfaction scores against text sentiment labels to identify inconsistencies in the source data.

## 🛡️ Business Logic & Guardrails

To ensure reliability, the system implements score-based overrides:

* **Score ≤ 2**: Automatically classified as **Negative**, triggering a "SORRY_2026" discount code.
* **Score ≥ 4**: Automatically classified as **Positive**, triggering the Trip Recommendation engine.
* **Score 3**: Relies on the LLM's nuanced sentiment analysis to decide the outcome.