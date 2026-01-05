from lstm_utils import get_ai_score

def analyze_text(text: str) -> float:
    prediction = get_ai_score(text)
    
    print(f"Input: {text[:30]}... -> Score: {prediction}")
    return prediction

if __name__ == "__main__":
    print("\n--- AI DETECTION TEST ---")
    
    
    blatant_ai_text = "As an AI language model, I do not have personal opinions or feelings. However, I can provide information based on the data I was trained on."
    
    result = analyze_text(blatant_ai_text)
    
    print(f"Metin: {blatant_ai_text[:50]}...")
    print(f"Result: {result}")
    
    if result > 0.5:
        print(f"Result: 🤖 AI (accuracy: %{result*100:.2f})")
    else:
        print(f"Result: 👤 HUMAN (accuracy: %{(1-result)*100:.2f})")