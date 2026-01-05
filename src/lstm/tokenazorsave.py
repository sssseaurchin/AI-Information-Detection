import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

print(">>> Data reading...")

dataset = pd.read_csv('data/AI_Human/AI_Human_cleaned.csv') 

text_col = dataset.columns[0]
label_col = dataset.columns[1]

# --- Cleaning (same as training) ---
print(">>> Cleaning...")
dataset = dataset.dropna(subset=[text_col, label_col])
dataset = dataset.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)

# --- SPLIT (random_state=0) ---
print(">>> Data splitting...")
x_train_text, x_test_text, y_train, y_test = train_test_split(
    dataset[text_col].astype(str),
    dataset[label_col],
    test_size=0.2,
    random_state=0,
    stratify=dataset[label_col]    
)

# --- TOKENIZER create ---
print(">>> Creating Tokenizer  (not model, just dictionary)...")
MAX_NB_WORDS = 20000      
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(x_train_text)

# --- SAVING ---
print(">>> Tokenizer saving...")

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("DONE! tokenizer.pickle created.")