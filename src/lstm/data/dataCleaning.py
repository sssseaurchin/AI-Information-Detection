import json 
import pandas as pd


# Fonksiyonlar -> veriyi alıp temizleyip veri_clean.csv haline kaydeder

def AI_Human() -> pd.DataFrame:
    path = "data/AI_Human.csv"
    df = pd.read_csv(path)
    df["generated"] = 1 - df["generated"]  # We need to flip !!! 
    df.to_csv("data/AI_Human_cleaned.csv", index=False)

AI_Human()

def humanVSAIJSONL() -> list: 
    jsonl_path = "dataall.jsonl"  # change path if needed
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        
        for line in f:
            obj = json.loads(line)

            # human answers
            for text in obj.get("human_answers", []): 
                rows.append({"text": text, "is_human": "1"})

            # chatgpt answers
            for text in obj.get("chatgpt_answers", []):
                rows.append({"text": text, "is_human": "0"})
        
        # print(rows) 
    df = pd.DataFrame(rows)
    df.to_csv("data/all_clean.csv", index=False)
