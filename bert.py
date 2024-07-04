import json
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification


# Load intents file
with open('intents2.json') as f:
    intents = json.load(f)

# Convert intents to DataFrame
data = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        data.append((pattern, intent['tag']))

df = pd.DataFrame(data, columns=['text', 'label'])

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

#Prebuild Model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['label'].unique()))

def convert_examples_to_tf_dataset(examples, labels, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []
    for example in examples:
        inputs = tokenizer.encode_plus(
            example,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])
    
    return tf.data.Dataset.from_tensor_slices((
        (input_ids, attention_masks),
        labels
    ))

train_dataset = convert_examples_to_tf_dataset(train_texts, train_labels, tokenizer)
test_dataset = convert_examples_to_tf_dataset(test_texts, test_labels, tokenizer)

train_dataset = train_dataset.shuffle(len(train_texts)).batch(16)
test_dataset = test_dataset.batch(16)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
from transformers import TFBertForSequenceClassification

# Attempt to save the pre-trained model
try:
    model.save('bert_intent_model', save_format='tf')
except MemoryError as e:
    print("Memory allocation error:", e)
 
# model.save('bert_intent_model', save_format='tf')
tokenizer.save('bert_intent_tokenizer')
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

