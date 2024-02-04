import streamlit as st
# For data manipulation
import pandas as pd
# For numerical operations
import numpy as np

import os
from openai import OpenAI


import json
# For TF-IDF vectorization and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
# Load the tarot card data from the JSON file
def load_tarot_data(json_file_path):
    with open(json_file_path, 'r') as file:
        json_data=json.load(file)
    return json_data

@st.cache_data
# because in the past analysis we realize that we only need the cards without null values
def extract_card_details(card_dict):
    # Extracting all known fields from the card
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
    
    
    # Handling list fields to convert them into string for easier display/storage
    for key in ['Fortune Telling', 'Keywords', 'Meanings Light', 'Meanings Shadow', 'Questions to Ask']:
        if isinstance(card_details[key], list):
            card_details[key] = '; '.join(card_details[key])
    
    return card_details



@st.cache_data
def preprocess_data(json_data):
    card_list= json_data['cards']
    extracted_details = [extract_card_details(card) for card in card_list]
    cards_df = pd.DataFrame(extracted_details)
    return cards_df
    

def select_one_card(df):
    selected_card = df.sample(n=1).reset_index(drop=True)
    selected_card['Period'] = ''
    return selected_card


def select_three_cards(df):
    selected_cards = df.sample(n=3).reset_index(drop=True)
    periods = ['Past', 'Present', 'Future']
    selected_cards['Period'] = periods
    
    return selected_cards



# Concatenate your text columns into a single text column for vectorization
def get_sim_matrix(df):
    combined_text = df['Keywords'] + '; ' + df['Meanings Light'] + '; ' + df['Meanings Shadow'] + '; ' + df['Questions to Ask']

    # Vectorizing  Text Data
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    # get similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


# get simliar cards
def get_most_similar_card(card_name, df, cosine_sim_matrix):
    card_index = df[df['Name'] == card_name].index[0]  # Get index of the card
    sim_scores = cosine_sim_matrix[card_index]  # Get similarity scores for the card
    sim_scores[card_index] = -1  # Ignore self-similarity by setting it to -1
    
    # Find the index of the card with the highest similarity score
    most_similar_index = np.argmax(sim_scores)
    
    # Directly create a DataFrame from the Series using .iloc and pass it within a list to maintain DataFrame structure
    most_similar_card = df.iloc[[most_similar_index]]
    
    return most_similar_card




def display_card_image_streamlit(card):
    image_path = f"cards/{card['Image'].iloc[0]}"  # Assuming 'Image' holds the filename
    st.image(image_path, caption=f"Behold the card: {card['Name'].iloc[0]}")


def display_three_cards_streamlit(selected_cards):
    periods = ['Past', 'Present', 'Future']
    
    for period, (_, card) in zip(periods, selected_cards.iterrows()):
        st.subheader(period)  # Display the period as a subheader
        display_card_image_streamlit(card)

def display_card_details_streamlit(card, period_label=''):
    if period_label:
        st.subheader(period_label)
    
    # Handling text fields that might be lists or strings
    fortune_telling = '; '.join(card['Fortune Telling']) if isinstance(card['Fortune Telling'].iloc[0], list) else card['Fortune Telling'].iloc[0]
    keywords = '; '.join(card['Keywords']) if isinstance(card['Keywords'].iloc[0], list) else card['Keywords'].iloc[0]
    meanings_light = '; '.join(card['Meanings Light']) if isinstance(card['Meanings Light'].iloc[0], list) else card['Meanings Light'].iloc[0]
    meanings_shadow = '; '.join(card['Meanings Shadow']) if isinstance(card['Meanings Shadow'].iloc[0], list) else card['Meanings Shadow'].iloc[0]
    questions_to_ask = '; '.join(card['Questions to Ask']) if isinstance(card['Questions to Ask'].iloc[0], list) else card['Questions to Ask'].iloc[0]


    st.write(f"**Name:** {fortune_telling}")
    st.write(f"**Suit:** {keywords}")
    st.write(f"**Fortune Telling:** {meanings_light}")
    st.write(f"**Keywords:** {meanings_shadow}")
    st.write(f"**Meanings Light:** {meanings_light}")
    st.write(f"**Meanings Shadow:** {meanings_shadow}")
    st.write(f"**Questions to Ask:** {questions_to_ask}")
    
    # Display the image
    image_path=f"cards/{card['Image'].iloc[0]}"
    st.image(image_path)




file_path = 'tarot-images.json'
data = load_tarot_data(file_path)
dff = preprocess_data(data)
cosine_sim = get_sim_matrix(dff)

# Initialize 'selected_card' in session state if it's not already present
if 'selected_card' not in st.session_state:
    st.session_state.selected_card = pd.DataFrame()  # An empty DataFrame as a placeholder

# Display a button for the user to select a card
if st.button('Select a Card for Today', key='select_one_card'):
    st.session_state.selected_card = select_one_card(dff)  # Update the session state
    st.write("The selected card for today you is:")
    display_card_details_streamlit(st.session_state.selected_card, period_label='')

# Check if a card has been selected and 'selected_card' is not empty
if not st.session_state.selected_card.empty:
    card_name_str = st.session_state.selected_card['Name'].iloc[0]
    most_similar_card = get_most_similar_card(card_name_str, dff, cosine_sim)
    st.write("\nThe most similar card is:")
    display_card_details_streamlit(most_similar_card)








from openai import OpenAI
import streamlit as st

st.title("ChatGPT-like clone")

client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

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




