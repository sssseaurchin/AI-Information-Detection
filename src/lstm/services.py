import joblib

try:
    _vectorizer = joblib.load("tfidf_vectorizer.joblib")
    _model = joblib.load("logreg_model.joblib")
except Exception as e:
    _vectorizer = None
    _model = None


def analyze_text(text: str) -> float: 
    if _vectorizer is None or _model is None:
        raise RuntimeError("Model or vectorizer not loaded. Train them first.")
    
    X = _vectorizer.transform([text])

    probs = _model.predict_proba(X)[0]  # list/array-like
    return float(max(probs))            # confidence of predicted class
