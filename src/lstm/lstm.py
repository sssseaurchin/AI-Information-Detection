import numpy as np
import pandas as pd
import joblib  # NEW
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utility import TimeTracker

timer = TimeTracker()
# dataset = pd.read_csv('data/Human ChatGPT Comparison Corpus (HC3)/all_clean.csv') # KÜÇÜK
dataset = pd.read_csv('data/AI_Human/AI_Human_cleaned.csv') # Büyük

text_col = dataset.columns[0]
label_col = dataset.columns[1]

dataset = dataset.dropna(subset=[text_col, label_col])
dataset = dataset.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)

x_train_text, x_test_text, y_train, y_test = train_test_split(
    dataset[text_col].astype(str),
    dataset[label_col],
    test_size=0.2,
    random_state=0,
    stratify=dataset[label_col]
)

set_train = set(x_train_text)
set_test = set(x_test_text)
print("AAAAAAAAAAA: ", len(set_train & set_test))  # ortak metin sayısı train ve test arasında

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
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))
print("Total time:", timer.get_elapsed_time())

# NEW: save vectorizer + model (you need both to use it later)
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(clf, "logreg_model.joblib")

# (optional) NEW: save raw weights too
joblib.dump(
    {
        "coef": clf.coef_,
        "intercept": clf.intercept_,
        "classes": clf.classes_,
        "vocab": vectorizer.vocabulary_,
        "idf": vectorizer.idf_,
    },
    "logreg_weights.joblib"
)
