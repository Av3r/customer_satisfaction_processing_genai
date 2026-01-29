# Overview
Customer Satisfaction Processing 

# Prerequisites
1. Creating a dedicated virtual environment (.venv) is recommended
2. Install everything listed in requirements.txt ```pip install -r requirements.txt```
3. Create and fill in the .evn file. (Check .env_template)
   1. OPEN_AI key is required
   2. LANGSMITH configuration is recommented but not mandatory
4Spacy installation (if not installed automatically): ```python -m spacy download en_core_web_md```

# How it works
## Initialization
**Create instance of LLM wrapper:**
- model: gpt-40-mini
- temperature: 0 (deterministic - expected same or similar answer every time)

### Initialize Embedding Model  
Translates input text into numbers (vectors).  
Embeddings are required to search through documents to find relevant information before answering a question.
- model: text_embedding-3-small (speed, low cost, good performance)

### Load the Name Entity Recognition (NER) Model
It uses spacy which may require manual installation.  
- model: en_core_web_md
NER is used for traditional NLP tasks like extracting trip_ref_id, city country, activities or details

### Load the Data
Read the JSON with trip details:    
```country, city, start date, count of days, cost in EUR, extra activities, trip details```

### Normalize and ID Creation
- **Unique ID:** It assigns a trip_ref_id based on the row index. This is critical for the AI to "point" to a specific trip later.
- **Lowercasing:** It creates lowercase copies of City and Country names. This ensures that searching for "PARIS" matches "Paris" without issues.

### Create the "Searchable" Text (Context)
- **The "Blob":** This combines all the important information about a trip (Location, Activities, Description) into a single string called rag_content.
- **Why?** The embedding model needs one cohesive text block to understand the "meaning" of the trip. The resulting text looks like:
```
Trip ID 5: Trip to Paris, France. Activities: Museum, Wine Tasting. Details: A lovely 5-day tour...
```

### Generate Vectors (Indexing)
Convert each trip description into a vector using the helper function

### Summary of the Output
At the end of this step, you have a table where every trip has:   
1. **Metadata:** Original details (Price, Duration, etc.). 
2. **Search Tags:** Lowercase cities/countries. 
3. **Vector:** A mathematical coordinate that represents the "meaning" of that trip.

### Construct Prompt Template
Instruct LLM on how to react, how to assess the review and how to generate the response 

### Build Runnable chain
1. Prepare and provide prompt
2. Ask LLM for assessment
3. Parse the response

### Location Extraction Function (NER)
Extract GPEs (Geopolitical Entities) - countries, cities, states from parsed LLM responses

### Find three the best trips
1. NER hard filter (match by city or country)
2. RAG semantic search (rank by embedding similarity)

If NER finds 3 or more recommendations skip RAG
Else use RAG to find location based on the semantic meaning i.e ```great sushi ---> Tokyo```

### Final decision
1. Compare the AI scoring with score provided by the clients and adjust it accordingly
2. Suggest RECOMMENDATION for a positive feedback and DISCOUNT for a negative one

**RECOMMENDATION** - details of three different trips the customer can be interested with   
**DISCOUNT** - provide discount code for the unpleasant customer 

### Results Visualisation
#### Confusion matrix
This chart measures accuracy. It answers: "How often did the AI's predicted sentiment match the actual ground truth?"

#### Recommendation source
This chart measures strategy. It answers: "Are we finding trips because of specific locations (NER) or general vibes (RAG)?"

## Helper functions
### get_embedding(text: str)   
- **Clean**: It replaces newline characters (\n) with spaces. This is a best practice because newlines can sometimes confuse embedding models or alter the semantic meaning. 
- **Convert**: It uses the embeddings_model (loaded in Step 1) to turn the text string into a list of numbers (a vector).

### prepare_trips_engine
Reads JSON file and returns Pandas DataFrame

## Enumerates
- **Strict output rules** - positive, negative, neutral
- Format response scheme: sentiment, summary

# Interactive Demo
Running ````run_manual_demo```` will open an interactive chat session in the terminal.   
You can provide review text and scoring to check how the application behaves.   

