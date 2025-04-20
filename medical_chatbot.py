import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
# Sets up the Streamlit web application with a title, icon, and layout
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥",
    layout="centered"
)

# Display the main title and description of the application
st.title("🏥 Medical Chatbot")
st.markdown("Ask me any health-related question, and I'll try to help you.")

# Text preprocessing function
def preprocess_text(text):
    """
    Prepares text for analysis by standardizing format and removing noise
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned and standardized text
    """
    # Handle non-string inputs gracefully
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase to standardize text
    text = text.lower()
    
    # Remove special characters, keeping only letters and spaces
    # This helps to focus on the semantic content rather than symbols
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace to standardize spacing
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Fast medical chatbot implementation using TF-IDF and cosine similarity
class FastMedicalChatbot:
    def __init__(self, min_df=1, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize a TF-IDF based chatbot for medical queries
        
        Args:
            min_df (int): Minimum document frequency for TF-IDF - filters out rare terms
            max_features (int): Maximum vocabulary size to control memory usage
            ngram_range (tuple): Range of word sequences to consider, helps capture phrases
        """
        # Initialize TF-IDF vectorizer with customizable parameters
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_features=max_features,
            ngram_range=ngram_range,
            analyzer='word',
        )
        self.df = None
        self.tfidf_matrix = None
        
    def train(self, df, text_column='Sentence'):
        """
        Fit the vectorizer and compute TF-IDF matrix from training data
        
        Args:
            df (DataFrame): Dataset containing questions/texts and responses
            text_column (str): Name of the column containing the input texts
            
        Returns:
            self: Returns the instance for method chaining
        """
        self.df = df
        
        # Preprocess all texts in the dataset for consistency
        texts = df[text_column].apply(preprocess_text).tolist()
        
        # Fit and transform the texts to create the TF-IDF matrix
        # This creates the knowledge base for answering questions
        print("Fitting TF-IDF vectorizer...")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        return self
    
    def get_most_similar(self, query, threshold=0.1):
        """
        Find the most similar question/text to the user query using cosine similarity
        
        Args:
            query (str): User input question
            threshold (float): Minimum similarity score to consider a match valid
            
        Returns:
            dict or None: Information about the best match if above threshold, None otherwise
        """
        # Preprocess query using the same pipeline as training data
        processed_query = preprocess_text(query)
        
        # Transform query to TF-IDF vector space
        query_vector = self.vectorizer.transform([processed_query])
        
        # Compute cosine similarity between query and all texts in the database
        # This measures how similar the query is to each entry in our knowledge base
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Find best match - the text with highest similarity score
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Return match info if it exceeds our confidence threshold
        if best_score >= threshold:
            return {
                'index': best_idx,
                'similarity': best_score,
                'question': self.df.iloc[best_idx]['Sentence']
            }
        else:
            # Return None if no sufficiently similar text was found
            return None
    
    def get_keywords(self, query, max_keywords=5):
        """
        Extract the most important keywords from the query based on TF-IDF weights
        
        Args:
            query (str): User input question
            max_keywords (int): Maximum number of keywords to extract
            
        Returns:
            list: Top keywords from the user query
        """
        # Preprocess query for consistency
        processed_query = preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        # Get feature names (words/tokens) from the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get keywords based on highest TF-IDF values
        # Higher TF-IDF values indicate more important/distinctive terms
        dense_vector = query_vector.toarray().flatten()
        indices = dense_vector.argsort()[-max_keywords:][::-1]
        
        # Extract the feature names for these indices, filtering out zero weights
        keywords = [feature_names[idx] for idx in indices if dense_vector[idx] > 0]
        
        return keywords

def load_data():
    """
    Load and preprocess the medical dataset from CSV
    
    Returns:
        DataFrame: Processed dataset with standardized column names
    """
    try:
        # Attempt to load the medical dataset CSV file
        df = pd.read_csv("ai-medical-chatbot.csv")
        
        # Check if the dataframe has specific column structure and adapt if needed
        if 'Description' in df.columns and 'Patient' in df.columns and 'Doctor' in df.columns:
            # Map columns to match our model's expected format
            # 'Description' contains questions, 'Doctor' contains answers
            df = pd.DataFrame({
                'Sentence': df['Description'],
                'Sentiment': df['Doctor']
            })
            
        elif 'Sentence' not in df.columns or 'Sentiment' not in df.columns:
            # Try to fix column headers if needed - assumes first column is question, second is answer
            df.columns = ['Sentence', 'Sentiment']
        
        print(f"Successfully loaded medical dataset with {len(df)} entries")
        
    except Exception as e:
        # Handle errors gracefully and provide feedback
        print(f"Error loading dataset: {e}")
        # Create a placeholder dataset with an error message
        df = pd.DataFrame({
            'Sentence': ["I'm unable to load the medical dataset. Please make sure it's in the correct location."],
            'Sentiment': ["I apologize, but I couldn't load my knowledge base. Please seek medical advice from a professional."]
        })
    
    return df

# Load and cache data to avoid reloading on every interaction
@st.cache_data
def initialize_data():
    """
    Load the dataset with caching for performance
    
    Returns:
        DataFrame: Medical QA dataset
    """
    # Load the dataset
    df = load_data()
    return df

# Initialize and cache the model to avoid retraining on every interaction
@st.cache_resource
def initialize_model(df):
    """
    Initialize and train the chatbot model with caching
    
    Args:
        df (DataFrame): Dataset to train on
        
    Returns:
        FastMedicalChatbot: Trained chatbot model
    """
    # Initialize model with customized parameters for medical text
    # Using wider n-gram range (1-3) to capture medical phrases
    model = FastMedicalChatbot(min_df=1, max_features=5000, ngram_range=(1, 3))
    model.train(df)
    
    return model

# Main application function
def main():
    """
    Main application logic for the Streamlit medical chatbot
    """
    # Initialize session state to store chat history between interactions
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load data and model with a loading indicator
    with st.spinner("Loading medical knowledge..."):
        df = initialize_data()
        model = initialize_model(df)
    
    # Display all previous chat messages from the session history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.chat_message('user').write(message['content'])
        else:
            st.chat_message('assistant').write(message['content'])

    # Get user input with Streamlit's chat interface
    user_question = st.chat_input("Type your health question here...")
    
    # Process new user input if available
    if user_question:
        # Add user message to chat history for persistent display
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question
        })
        
        # Display the new user message
        st.chat_message('user').write(user_question)
        
        # Process the question and generate a response
        with st.spinner("Finding answer..."):
            # Find the most similar question in our database
            result = model.get_most_similar(user_question, threshold=0.1)
            
            # Extract keywords for debugging and transparency
            keywords = model.get_keywords(user_question)
            keyword_text = ', '.join(keywords) if keywords else "general semantic matching"
            
            # Generate response based on match results
            if result:
                # If match found, use the corresponding answer from the dataset
                answer_idx = result['index']
                answer = df.iloc[answer_idx]['Sentiment']
                response = answer
                
                # Log the matching details for debugging/analysis
                print(f"Query: {user_question}")
                print(f"Matched with: {df.iloc[answer_idx]['Sentence']}")
                print(f"Similarity: {result['similarity']:.4f}")
                print(f"Keywords: {keyword_text}")
            else:
                # Fallback response when no good match is found
                response = "I'm sorry, I don't have enough information to answer that medical question accurately. Please consult a healthcare professional for personalized advice."
                print(f"No match found for: {user_question}")
        
        # Display the chatbot's response
        st.chat_message('assistant').write(response)
        
        # Add the response to chat history for persistence
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response
        })

    # Display information and medical disclaimer in an expandable section
    with st.expander("About this Chatbot"):
        st.write("""
        This medical chatbot uses advanced text analysis to match your questions with answers 
        from a curated medical knowledge base.
        
        **Disclaimer**: This chatbot provides general information and is not a substitute 
        for professional medical advice, diagnosis, or treatment. Always seek the advice 
        of your physician or other qualified health provider with any questions you may 
        have regarding a medical condition.
        """)

# Entry point of the application
if __name__ == "__main__":
    main()
