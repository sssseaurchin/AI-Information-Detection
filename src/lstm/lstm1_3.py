import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utility import TimeTracker

# --- TENSORFLOW IMPORTS ---
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

# TIMER
timer = TimeTracker()





# --- 1. DATA LOADING ---
print(">>> Data loading...")
# dataset = pd.read_csv('data/Human ChatGPT Comparison Corpus (HC3)/all_clean.csv') 
dataset = pd.read_csv('data/AI_Human/AI_Human_cleaned.csv') 

text_col = dataset.columns[0]
label_col = dataset.columns[1]






# --- 2. CLEANING ---
print(">>> CLEANING...")
dataset = dataset.dropna(subset=[text_col, label_col])
dataset = dataset.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)

print(f"DATA NUMBER: {len(dataset)}")





# --- 3. TRAIN-TEST (SPLIT) ---
print(">>> DATA SPLITS AS Train/Test...")

x_train_text, x_test_text, y_train, y_test = train_test_split(
    dataset[text_col].astype(str),
    dataset[label_col],
    test_size=0.2,
    random_state=0,
    stratify=dataset[label_col]    
)

# LEAKAGE CONTROL
set_train = set(x_train_text)
set_test = set(x_test_text)
print("AAAAAAAAAAA (Common text between test and train): ", len(set_train & set_test))





# --- 4. (TOKENIZATION) ---

print(">>> Tokenization BEGINNING...")

MAX_NB_WORDS = 20000      # common 20k words
MAX_SEQUENCE_LENGTH = 200 # length
EMBEDDING_DIM = 100       # word vector size

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
# DİKKAT: Tokenizer JUST FOR TRAINING (Fit)
tokenizer.fit_on_texts(x_train_text)

# convert text to numbers
x_train_seq = tokenizer.texts_to_sequences(x_train_text)
x_test_seq = tokenizer.texts_to_sequences(x_test_text)

# (Padding)
x_train_pad = pad_sequences(x_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
x_test_pad = pad_sequences(x_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

print(f"Eğitim verisi boyutu (Shape): {x_train_pad.shape}")
print(f"Test verisi boyutu (Shape): {x_test_pad.shape}")





# --- 5. (LSTM) ---
print(">>> CREATIN MODEL...")

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())





# --- 6. (TRAINING) ---
print(">>> BEGINNING TRAINING...")

batch_size = 64
epochs = 5 

history = model.fit(x_train_pad, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    validation_split=0.1, 
                    verbose=1)





# --- 7. (PREDICT) ---
print(">>> (PREDICT)...")


y_pred_probs = model.predict(x_test_pad)


y_pred = (y_pred_probs > 0.5).astype(int)

print("-" * 50)
print("RESULTS")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("-" * 50)
print("Total time:", timer.get_elapsed_time())




# --- 8. (SAVE) ---



import pickle

# 1. Save Model
print(">>> Saving model...")
model.save('ai_detector_model.keras')
print("✅ Model file (.keras) created.")

# 2. Save Tokenizer
print(">>> saving Tokenizer...")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("✅ Tokenizer file (.pickle) created.")
