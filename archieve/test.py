import json
import pandas as pd
import random


# Load the tarot card data from the JSON file
file_path='tarot-images.json'


with open(file_path, 'r') as file:
    tarot_data = json.load(file)



# Display the structure of the loaded tarot data for an overview
tarot_data_keys = list(tarot_data.keys())
sample_card = next(iter(tarot_data.values()))

# Showing the structure and a sample card to understand the data format
tarot_data_structure = {
    "Total Cards": len(tarot_data),
    "Data Structure Keys": tarot_data_keys,
    "Sample Card": sample_card
}



# Extracting and inspecting the data under the 'cards' key
cards_data = tarot_data['cards']

############################################################

def matchPattern(user_input):
    # Simplified intent matching based on keywords in user input
    if "tarot reading" in user_input.lower():
        return "tarot_reading"
    elif "learn about" in user_input.lower():
        return "learn_about_card"
    else:
        return "unknown"

def getResponse(intent):
    # Generate responses based on the identified intent
    if intent == "tarot_reading":
        card = random.choice(['The Fool', 'The Magician', 'The High Priestess'])  # Example card selection
        return f"Your card is {card}. (Further reading details would go here.)"
    elif intent == "learn_about_card":
        return "Please specify which card you want to learn about by typing 'learn about [card name]'."
    else:
        return "I'm not sure how to respond to that. Can you try asking something else?"

def main():
    print("Hello! I am a tarot chatbot. Type 'q' to end our conversation.")
    
    while True:
        user_input = input("You: ")
        
        # Check for the quit command
        if user_input.lower() in ['q']:
            print("Chatbot: Goodbye!")
            break
        
        intent = matchPattern(user_input)
        response = getResponse(intent)
        print("Chatbot:", response)

# Start the chatbot
main()

