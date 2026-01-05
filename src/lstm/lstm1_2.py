import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utility import TimeTracker
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline


timer = TimeTracker()
# dataset = pd.read_csv('data/Human ChatGPT Comparison Corpus (HC3)/all_clean.csv') 
dataset = pd.read_csv('data/AI_Human/AI_Human_cleaned.csv') # Büyük

text_col = dataset.columns[0]
label_col = dataset.columns[1]
 
 #-----------------------------------CLEANING-----------------------------------------
dataset = dataset.dropna(subset=[text_col, label_col])
 
dataset = dataset.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)
 
 #-----------------------------------K-FOLD-----------------------------------------

x_all = dataset[text_col].astype(str).values
y_all = dataset[label_col].values

pipeline = make_pipeline(
    TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words='english'
    ),
    LogisticRegression(max_iter=1000)
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv_scores = cross_val_score(pipeline, x_all, y_all, cv=cv, scoring="accuracy")
print("KFOLD")
print("CV accuracies:", cv_scores)
print("CV mean accuracy:", cv_scores.mean())
print("CV std:", cv_scores.std())
print("-" * 50)



# dataset = dataset.drop_duplicates(subset=[text_col]).reset_index(drop=True)
 
x_train_text, x_test_text, y_train, y_test = train_test_split(
    dataset[text_col].astype(str),
    dataset[label_col],
    test_size=0.2,
    random_state=0,
    stratify=dataset[label_col]    
)
 
set_train = set(x_train_text)
set_test = set(x_test_text) 



print("AAAAAAAAAAA: ", len(set_train & set_test) )


vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words='english'
)

x_train = vectorizer.fit_transform(x_train_text)
x_test = vectorizer.transform(x_test_text)



clf = LogisticRegression(max_iter=1000)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("PREDICT ")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))
print("Total time:", timer.get_elapsed_time())
