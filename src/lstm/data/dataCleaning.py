import json 
import pandas as pd


# Fonksiyonlar -> veriyi alıp temizleyip veri_clean.csv haline kaydeder

def AI_Human() -> pd.DataFrame:
    path = "data/AI_Human.csv/AI_Human.csv"
    df = pd.read_csv(path)
    df["generated"] = 1 - df["generated"]  # We need to flip !!! 
    df.to_csv("data/AI_Human/AI_Human_cleaned.csv", index=False)


def humanVSAIJSONL() -> list: 
    jsonl_path = "data/Human ChatGPT Comparison Corpus (HC3)/all.jsonl"  # change path if needed
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        
        for line in f:
            obj = json.loads(line)

            # human answers
            for text in obj.get("human_answers", []): 
                rows.append({    "text": text,    "is_human": "1"})

            # chatgpt answers
            for text in obj.get("chatgpt_answers", []):
                rows.append({    "text": text,    "is_human": "0"})
        
        # print(rows) 
    df = pd.DataFrame(rows)
    df.to_csv("data/Human ChatGPT Comparison Corpus (HC3)/all_clean.csv", index=False)
 

"""
def save_df_to_mega(new_df : pd.DataFrame):
    MEGALADON = "data/megaladon.csv" 

    # if file exists, load and append safely 
    existing_df = pd.read_csv(MEGALADON)

    # combine + remove duplicates
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(inplace=True)

    combined_df.to_csv(MEGALADON, index=False) 

def save_list_to_mega(data:list):  # kinda dumb to have this
    new_df = pd.DataFrame(data)
    save_df_to_mega(new_df)

"""
