import json
import random
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pickle

# Load the pre-trained model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('bert_intent_model')
tokenizer = BertTokenizer.from_pretrained('bert_intent_tokenizer')
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Load intents file
with open('intents2.json') as f:
    intents = json.load(f)

# Prediction functions
def predict_intent(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_class = np.argmax(logits, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]

def get_response(intent, intents_json):
    for i in intents_json['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

def chatbot_response(text):
    intent = predict_intent(text)
    response = get_response(intent, intents)
    return response

# Run the chatbot
print("Start talking with the bot (type 'quit' to stop)!")
while True:
    message = input("")
    if message.lower() == "quit":
        break
    response = chatbot_response(message)
    print(response)
