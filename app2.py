import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_tarot_data(json_file_path):
    with open(json_file_path, 'r') as file:
        json_data=json.load(file)
    return json_data

@st.cache_data
def extract_card_details(card_dict):
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

@st.cache_data
def preprocess_data(json_data):
    card_list = json_data['cards']
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

def get_sim_matrix(df):
    combined_text = df['Keywords'] + '; ' + df['Meanings Light'] + '; ' + df['Meanings Shadow'] + '; ' + df['Questions to Ask']
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_most_similar_card(card_name, df, cosine_sim_matrix):
    card_index = df[df['Name'] == card_name].index[0]
    sim_scores = cosine_sim_matrix[card_index]
    sim_scores[card_index] = -1
    most_similar_index = np.argmax(sim_scores)
    most_similar_card = df.iloc[[most_similar_index]]
    return most_similar_card

def display_card_image_streamlit(card):
    image_path = f"cards/{card['Image'].iloc[0]}"
    st.image(image_path, caption=f"Behold the card: {card['Name'].iloc[0]}")

def display_card_details_streamlit(card, period_label=''):
    if period_label:
        st.subheader(period_label)
    
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
    
    image_path = f"cards/{card['Image'].iloc[0]}"
    st.image(image_path)

file_path = 'tarot-images.json'
data = load_tarot_data(file_path)
dff = preprocess_data(data)
cosine_sim = get_sim_matrix(dff)

st.title("Tarot Card Reader Chatbot")

st.title("ChatGPT-like Clone")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Bot: What would you like to ask?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "Select a Card" in prompt:
        selected_card = select_one_card(dff)
        st.subheader("Selected Card:")
        display_card_with_questions(selected_card)

        card_name_str = selected_card['Name'].iloc[0]
        most_similar_card = get_most_similar_card(card_name_str, dff, cosine_sim)
        st.subheader("Most Similar Card:")
        display_card_with_questions(most_similar_card)

        user_input = st.chat_input("Bot: What do you think about the selected card?")
        st.session_state.messages.append({"role": "user", "content": user_input})

        extract_information_and_explain(user_input, selected_card, most_similar_card)

        user_response = st.radio("Bot: Would you like to accept questions from the cards? (Yes/No)", ["Yes", "No"])

        if user_response == "Yes":
            st.write("Bot: Great! Let's explore the questions from the cards:")
            questions_to_ask = selected_card['Questions to Ask'].split(';')

            for question in questions_to_ask:
                user_answer = st.text_input(f"Bot: {question.strip()}")
                st.session_state.messages.append({"role": "user", "content": user_answer})
                # Your logic to process user answers here

            st.write("Bot: Now, summarizing your answers and providing an explanation:")
            # Your logic to summarize user answers and provide an explanation here

        st.write("Bot: Chat session ended.")
