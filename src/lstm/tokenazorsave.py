import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer


dataset = pd.read_csv('data/AI_Human/AI_Human_cleaned.csv') 

text_col = dataset.columns[0]
label_col = dataset.columns[1]
 
print("Cleaning...")
dataset = dataset.dropna(subset=[text_col, label_col])
dataset = dataset.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)
 
print("Data splitting...")
x_train_text, x_test_text, y_train, y_test = train_test_split(
    dataset[text_col].astype(str),
    dataset[label_col],
    test_size=0.2,
    random_state=0,
    stratify=dataset[label_col]    
)
 
print("Creating Tokenizer  (not model, just dictionary)...")
MAX_NB_WORDS = 20000      
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(x_train_text)
 
print("Tokenizer saving...")

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("DONE! tokenizer.pickle created.")



""" utility.py

# 23.12.2025; Happy new years everyone
import time

class TimeTracker:
    def __init__(self): # Time is started after class initialization
        self.start_time = time.time()

    def get_elapsed_time(self):
        return round(time.time() - self.start_time, 2)
    
    def reset_timer(self):
        self.start_time = time.time()
    

if __name__ == "__main__":
    timer = TimeTracker()

    time.sleep(2)

    print(f"1. Time: {timer.get_elapsed_time()} seconds")
    timer.reset_timer()
    time.sleep(3)

    print(f"2. Time after reset: {timer.get_elapsed_time()} seconds")
"""
