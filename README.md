# Medical Chatbot

A Streamlit application that provides automated responses to health-related questions using TF-IDF vectorization and cosine similarity.

## Overview

This Medical Chatbot application matches user health queries against a curated dataset of medical questions and answers. It uses natural language processing techniques to find the most semantically similar questions in its knowledge base and returns the corresponding expert-provided answers.

## Features

- Interactive chat interface using Streamlit
- Fast query matching using TF-IDF vectorization
- Keyword extraction to improve transparency
- Session persistence to maintain conversation history
- Clear medical disclaimers and information

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Shafaat-Afridi/medical-chatbot.git
   cd medical-chatbot
   ```

2. Install required dependencies:
   ```
   pip install streamlit pandas numpy scikit-learn
   ```

3. Prepare your dataset:
   Place your medical QA dataset in the same directory as the script, named `ai-medical-chatbot.csv`. The file should contain at least two columns:
   - Questions/descriptions (labeled as 'Sentence' or 'Description')
   - Expert answers (labeled as 'Sentiment' or 'Doctor')

## Usage

Run the application using Streamlit:
```
streamlit run medical_chatbot.py
```

The application will be available at http://localhost:8501 by default.

## Dataset Format

The application expects a CSV file with the following structure:

Option 1:
```
Sentence,Sentiment
"What are the symptoms of the flu?","Flu symptoms typically include fever, chills, cough..."
```

Option 2:
```
Description,Patient,Doctor
"What are the symptoms of the flu?","I've been feeling feverish and achy.","Flu symptoms typically include fever, chills, cough..."
```

The application will attempt to adapt to either format.

## How It Works

1. The system preprocesses all texts in the medical knowledge base
2. User queries are preprocessed using the same pipeline
3. TF-IDF vectors are generated for both the knowledge base and user queries
4. Cosine similarity is calculated between the query and all entries
5. The answer corresponding to the most similar question is returned if similarity exceeds a threshold
6. If no sufficient match is found, a fallback response is provided

## Limitations

- Responses are limited to information in the knowledge base
- Cannot diagnose medical conditions or provide personalized medical advice
- May not understand highly technical medical terminology not present in training data
- Not a substitute for professional medical consultation

## Future Improvements

- Enhanced preprocessing for medical terminology
- Integration with medical ontologies
- Support for multi-turn medical conversations
- Addition of explanatory references for answers

## Disclaimer

This chatbot provides general information and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
