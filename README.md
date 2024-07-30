# Calculating and Reporting Metrics of the RAG Pipeline of Wedding Music Recommendation Chatbot

This repository contains the code and documentation for the Wedding Music Recommendation Chatbot version 1, which utilizes a Retrieval-Augmented Generation (RAG) pipeline to recommend wedding music. The chatbot is designed to assist users in finding suitable music for various wedding moments.

## Wedding Music Recommendation Chatbot version 1

https://github.com/YueWang19/Adaptive-recommendation-chatbot-with-RAG-and-vector-database

This repository contains the version 1 chatbot and the calculation function, you can clone and run to play

## Report

The detailed report on "Calculating and Reporting Metrics of the RAG Pipeline in Wedding Music Recommendation Chatbot" can be found in the `Calculating_and_Reporting_Metrics_of_the_RAG_Pipeline_in_Wedding_Music_Recommendation_Chatbot.pdf` file. The report includes the following sections:

1. Performance Metrics Calculation
   - Retrieval Metrics
   - Generation Metrics
   - Latency
2. Methods to Improve Metrics
3. Challenges Faced and Addressed

## How to Run and Test the Code

This repository contains the version 2 chatbot and the calculation function, you can clone and run to play

### Prerequisites

- Python 3.10 or higher
- Pip (Python package installer)
- Pinecone API key
- OpenAI API key

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/YueWang19/Calculating-and-Reporting-Metrics-of-the-RAG-Pipeline.git
   cd Calculating-and-Reporting-Metrics-of-the-RAG-Pipeline
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables for API keys:

   ```bash
   export PINECONE_API_KEY='your_pinecone_api_key'
   export OPENAI_API_KEY='your_openai_api_key'
   export PINECONE_CLOUD='your_pinecone_cloud_provider'  # default is 'aws'
   export PINECONE_REGION='your_pinecone_region'        # default is 'us-east-1'
   ```

### Running the Chatbot

1. Run the Python script to set up the Pinecone index and prepare the documents for search:

   ```bash
   python app.py
   ```

2. Run the Streamlit app to start the chatbot:

   ```bash
   streamlit run app.py
   ```

3. Open your web browser and go to `http://localhost:8501` to interact with the chatbot.

### Testing the Code

The `ragchatbotv2.py` script includes the main logic for the chatbot, including:

- Setting up Pinecone and defining embeddings
- Preparing the document for search
- Creating the vector store
- Implementing the chatbot's retrieve function
- Calculating and printing retrieval contexts and latency

To test the chatbot, you can use the example queries provided in the `ragchatbotv2.py` script:

```python
query1 = "Can you recommend some songs for a beach wedding ceremony?"
query2 = "What are some classical music suitable for the first dance?"
query3 = "What's the most popular song 2023?"

# Retrieve contexts and generate answers
retrieve_context_of_v2(query1)
retrieve_context_of_v2(query2)
retrieve_context_of_v2(query3)

# Measure response time
measure_response_time_v2(query1)
measure_response_time_v2(query2)
measure_response_time_v2(query3)
```
