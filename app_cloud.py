# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Import the SentimentIntensityAnalyzer

# Tarot Card Functionality

# Function to load tarot card data from a JSON file
@st.cache_data
def load_tarot_data(json_file_path):
    """Load tarot data from a JSON file.
    
    Args:
        json_file_path: The path to the JSON file containing tarot card data.
    
    Returns:
        A dictionary with tarot card data.
    """
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)
    return json_data

# Function to extract details from each tarot card entry
@st.cache_data
def extract_card_details(card_dict):
    """Extract details from a single tarot card entry.
    
    Args:
        card_dict: A dictionary representing a single tarot card.
    
    Returns:
        A dictionary with extracted details from the tarot card.
    """
    card_details = {
        'Name': card_dict['name'],
        'Number': card_dict['number'],
        'Arcana': card_dict['arcana'],
        'Suit': card_dict['suit'],
        'Image': card_dict['img'],
        'Fortune Telling': card_dict['fortune_telling'],
        'Keywords': card_dict['keywords'],
        'Meanings Light': card_dict['meanings']['light'],
        'Meanings Shadow': card_dict['meanings']['shadow'],
        'Questions to Ask': card_dict.get('Questions to Ask', [])
    }
    for key in ['Fortune Telling', 'Keywords', 'Meanings Light', 'Meanings Shadow', 'Questions to Ask']:
        if isinstance(card_details[key], list):
            card_details[key] = '; '.join(card_details[key])
    return card_details

# Function to preprocess loaded JSON data into a pandas DataFrame
@st.cache_data
def preprocess_data(json_data):
    """Preprocess JSON data into a pandas DataFrame.
    
    Args:
        json_data: The loaded JSON data containing tarot cards.
    
    Returns:
        A pandas DataFrame with tarot card details.
    """
    card_list = json_data['cards']
    extracted_details = [extract_card_details(card) for card in card_list]
    cards_df = pd.DataFrame(extracted_details)
    return cards_df


# Function to randomly select one card from the DataFrame
def select_one_card(df):
    """Select one card randomly from the DataFrame.
    
    Args:
        df: A pandas DataFrame containing tarot card details.
    
    Returns:
        A pandas DataFrame with details of the selected card.
    """
    selected_card = df.sample(n=1).reset_index(drop=True)
    selected_card['Period'] = ''
    return selected_card

# Function to compute the similarity matrix based on TF-IDF
def get_sim_matrix(df):
    """Compute the similarity matrix of tarot cards based on TF-IDF of card details.
    
    Args:
        df: A pandas DataFrame containing tarot card details.
    
    Returns:
        A cosine similarity matrix.
    """
    combined_text = df['Keywords'] + '; ' + df['Meanings Light'] + '; ' + df['Meanings Shadow'] + '; ' + df['Questions to Ask']
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


# Function to find the most similar card based on the similarity matrix
def get_most_similar_card(card_name, df, cosine_sim_matrix):
    """Find the most similar card to the given card name based on cosine similarity.
    
    Args:
        card_name: The name of the card to find similarities for.
        df: A pandas DataFrame containing tarot card details.
        cosine_sim_matrix: The cosine similarity matrix.
    
    Returns:
        A pandas DataFrame with details of the most similar card.
    """
    card_index = df[df['Name'] == card_name].index[0]
    sim_scores = cosine_sim_matrix[card_index]
    sim_scores[card_index] = -1
    most_similar_index = np.argmax(sim_scores)
    most_similar_card = df.iloc[[most_similar_index]]
    return most_similar_card

# Function to display a tarot card image in Streamlit
def display_card_image_streamlit(card):
    """Display a tarot card image using Streamlit.
    
    Args:
        card: A pandas DataFrame with details of the card to display.
    """
    image_path = f"cards/{card['Image'].iloc[0]}"  
    st.image(image_path, caption=f"Behold the card: {card['Name'].iloc[0]}")

# Function to display tarot card details in Streamlit
def display_card_details_streamlit(card, period_label=''):
    """Display tarot card details using Streamlit.
    
    Args:
        card: A pandas DataFrame with details of the card to display.
        period_label: Optional label indicating the time period of the reading.
    """
    if period_label:
        st.subheader(period_label)
    
    # Handling text fields that might be lists or strings
    name=card['Name'].iloc[0]
    Suit=card['Suit'].iloc[0]
    fortune_telling = '; '.join(card['Fortune Telling']) if isinstance(card['Fortune Telling'].iloc[0], list) else card['Fortune Telling'].iloc[0]
    keywords = '; '.join(card['Keywords']) if isinstance(card['Keywords'].iloc[0], list) else card['Keywords'].iloc[0]
    meanings_light = '; '.join(card['Meanings Light']) if isinstance(card['Meanings Light'].iloc[0], list) else card['Meanings Light'].iloc[0]
    meanings_shadow = '; '.join(card['Meanings Shadow']) if isinstance(card['Meanings Shadow'].iloc[0], list) else card['Meanings Shadow'].iloc[0]
    questions_to_ask = '; '.join(card['Questions to Ask']) if isinstance(card['Questions to Ask'].iloc[0], list) else card['Questions to Ask'].iloc[0]

    st.write(f"**Name:** {name}")
    st.write(f"**Suit:** {Suit}")
    st.write(f"**Fortune Telling:** {fortune_telling}")
    st.write(f"**Keywords:** {keywords}")
    st.write(f"**Meanings Light:** {meanings_light}")
    st.write(f"**Meanings Shadow:** {meanings_shadow}")
    st.write(f"**Questions to Ask:** {questions_to_ask}")
    
    # Display the image
    image_path = f"cards/{card['Image'].iloc[0]}"
    st.image(image_path)

# ChatGPT-like Interaction Functionality

def chat_interface():
    """
    Defines the chat interface for interacting with GPT-3.5-turbo via Streamlit.
    This function sets up the chat interface, manages session state for messages,
    and handles the interaction logic.
    """

    # for local testinf
    #client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # for streamlit cloud deployment
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    st.title("Talk about your day with gpt3.5-turbo!")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
        # Check if there's a tarot reading result to share
    if 'tarot_reading_result' in st.session_state:
        tarot_result = st.session_state['tarot_reading_result']
        # Format the tarot card details into a string or a markdown message
        tarot_message = (
            "Could you help me to understand today's tarot card? Here are the details: /n"
            "Name: {name}, "
            "Number: {number}, "
            "Arcana: {arcana}, "
            "Suit: {suit}, "
            "Fortune Telling: {fortune_telling}, "
            "Keywords: {keywords}, "
            "Meanings Light: {meanings_light}, "
            "Meanings Shadow: {meanings_shadow}, "
            "Questions to Ask: {questions_to_ask} "
        ).format(
            name=tarot_result.get('Name', 'N/A'),
            number=tarot_result.get('Number', 'N/A'),
            arcana=tarot_result.get('Arcana', 'N/A'),
            suit=tarot_result.get('Suit', 'N/A'),
            fortune_telling=tarot_result.get('Fortune Telling', 'N/A'),
            keywords=tarot_result.get('Keywords', 'N/A'),
            meanings_light=tarot_result.get('Meanings Light', 'N/A'),
            meanings_shadow=tarot_result.get('Meanings Shadow', 'N/A'),
            questions_to_ask=tarot_result.get('Questions to Ask', 'N/A').replace('; ', ', ')
        )

        st.session_state.messages.append({"role": "user", "content": tarot_message})
    


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Sentiment Analysis Function
def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using VADER sentiment analysis.
    
    Args:
        text: The input text to analyze.
    
    Returns:
        A string indicating the overall sentiment ('positive' or 'negative').
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return "positive" if sentiment['compound'] >= 0 else "negative"

# Intent Recognition Function
def recognize_intent(text):
    """
    Recognizes the user's intent based on keywords in the input text.
    
    Args:
        text: The input text to analyze for intent.
    
    Returns:
        The recognized intent ('love', 'career', 'health', or 'general').
    """

    intents = {
        'love': ['love', 'relationship', 'partner'],
        'career': ['job', 'career', 'work'],
        'health': ['health', 'sick', 'ill']
    }
    text = text.lower()
    for intent, keywords in intents.items():
        if any(keyword in text for keyword in keywords):
            return intent
    return "general"


# Generate Tarot Reading
def get_tarot_reading(dff,mode,sentiment,intent):
    """
    Generates a personalized tarot reading based on the user's query.
    
    Args:
        dff: The DataFrame containing tarot card details.
        mode: The selected reading mode.
        sentiment: The overall sentiment of the user's query.
        intent: The recognized intent of the user's query.
    """
    st.write('\nGenerating your personalized tarot reading... 🔮\n')
    
    modes = [['Past', 'Present', 'Future'], ['Situation', 'Action', 'Outcome'], ['You', 'Partner', 'Relationship']]
    cards3 = random.sample(range(78), k=3)
    readings = []

    for i in range(3):
        card = dff.iloc[cards3[i]]
        reading = {
            'name': card['Name'],
            'image': card['Image'],
            'fortune_telling': random.choice(card['Fortune Telling'].split('; ')),
            'meaning': card['Meanings Light'] if sentiment == 'positive' else card['Meanings Shadow']
        }
        readings.append(reading)

        # Display card image
        st.image(f"cards/{reading['image']}", caption=f"Card {i+1}: {reading['name']}")
        st.write(f"**{modes[int(mode.split(':')[0])][i]} Reading:** {reading['fortune_telling']}")
        st.write(f"**Card Meaning:** {reading['meaning']}")

    # Additional logic based on 'intent' to display specific details
    if intent == 'love':
        # Display love-related details
        st.write("This is a love-related reading.")
    elif intent == 'career':
        # Display career-related details
        st.write("This is a career-related reading.")
    elif intent == 'health':
        # Display health-related details
        st.write("This is a health-related reading.")
    else:
        # Default case for general reading
        st.write("This is a general reading.")

# Main App 

def main():
    """
    Main function to run the Streamlit app. This function initializes the app,
    sets up the navigation, and controls the app mode for different functionalities:
    Tarot Card Reader, ChatGPT-like Interaction, and Personalized Tarot Reading.
    """
    # Set up the main title and sidebar navigation
    st.title("Tarot Card and ChatGPT-like Interaction")
    st.sidebar.title("Navigation")
   # Sidebar for selecting the app mode
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Tarot Card Reader", "ChatGPT-like Interaction","Personalized Tarot Reading"])
    # Load tarot card data and preprocess it
    file_path = 'tarot-images.json'
    data = load_tarot_data(file_path)
    dff = preprocess_data(data)

    # Tarot Card Reader Mode
    if app_mode == "Tarot Card Reader":
        # Calculate cosine similarity matrix for tarot cards
        cosine_sim = get_sim_matrix(dff)
        
        # Button to select a card for today and display its details
        if st.button('Select a Card for Today'):
            selected_card = select_one_card(dff)
            st.session_state.selected_card = selected_card
            st.write("The selected card for today is:")
            display_card_details_streamlit(selected_card, period_label='')
            # Find and display the most similar card
            card_name_str = st.session_state.selected_card['Name'].iloc[0]
            most_similar_card = get_most_similar_card(card_name_str, dff, cosine_sim)
            st.write("\nThe most similar card is:")
            display_card_details_streamlit(most_similar_card)

            

            # Storing selected card details in session state for potential further use
            st.session_state['tarot_reading_result'] = selected_card.to_dict(orient='records')[0]  

        

    # ChatGPT-like Interaction Mode
    elif app_mode == "ChatGPT-like Interaction":
        # Function call to handle chat interface logic
        chat_interface()
    
    elif app_mode == "Personalized Tarot Reading":
        # UI elements for personalized tarot reading options
        st.title("Personalized Tarot Reading")
        mode = st.radio("Choose a reading mode:", ["0 : Past-Present-Future", "1: Situation-Action-Outcome", "2: You-Partner-Relationship"])
        user_input = st.text_input("Enter your query:")
        # Analyze sentiment and intent of the user query
        sentiment = analyze_sentiment(user_input)
        intent = recognize_intent(user_input)
        
        # Button to generate and display the personalized tarot reading
        if st.button("Generate Reading"):
            get_tarot_reading(dff,mode,sentiment,intent)

# Entry point of the script
if __name__ == "__main__":
    main()



