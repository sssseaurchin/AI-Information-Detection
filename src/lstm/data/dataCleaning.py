import json 
import pandas as pd


# Fonksiyonlar -> veriyi alıp temizleyip veri_clean.csv haline kaydeder

def AI_Human() -> pd.DataFrame:
    path = "data/AI_Human.csv"
    df = pd.read_csv(path)
    df["generated"] = 1 - df["generated"]  # We need to flip !!! 
    df.to_csv("data/AI_Human_cleaned.csv", index=False)


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


def students_vs_ai() -> pd.DataFrame:
    path = "data/AI_Human.csv"
    out_path = "data/AI_Human_cleaned.csv"

    df = pd.read_csv(path)

    df = df.rename(columns={"Text": "text", "Label": "Generated"})

    # normalize
    labels = df["Generated"].astype(str).str.strip().str.lower()

    mapping = {"student": 0, "ai": 1}
    mapped = labels.map(mapping)

    # show/handle bad labels
    bad = df.loc[mapped.isna(), "Generated"].value_counts(dropna=False)
    if not bad.empty:
        raise ValueError(
            "Found labels not in {student, ai}. Examples/counts:\n"
            + bad.head(20).to_string()
        )

    df["Generated"] = mapped.astype("int32")

    # optional: remove empty texts
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)

    df.to_csv(out_path, index=False)
    return df



students_vs_ai()