import json 
import pandas as pd


# Fonksiyonlar -> veriyi alıp temizleyip veri_clean.csv haline kaydeder
# HUMAN = 0; AI = 1;    

DATA_FOLDER = "data/"

def AI_Human() -> pd.DataFrame:
    path = "data/AI_Human.csv"
    df = pd.read_csv(path)
    # df["generated"] = 1 - df["generated"] 
    df = df.rename(columns={
        "Text": "text", 
        "generated": "isGenerated"
    })
    df.to_csv("data/AI_Human_cleaned.csv", index=False)

def ai_vs_human_comparison_dataset() -> pd.DataFrame:
    path = "data/ai-vs-human-comparison-dataset.csv"
    df = pd.read_csv(path)

    df = df[["text", "label"]]

    labels = df["label"].astype(str).str.strip().str.lower()

    mapping = {"human": 0, "ai": 1}
    df["label"] = labels.map(mapping)

    df = df.rename(columns={"label": "isGenerated"})

    df = df.dropna(subset=["isGenerated"])
    df["isGenerated"] = df["isGenerated"].astype(int)
    df.to_csv("data/ai_vs_human_comparison_dataset_cleaned.csv", index=False)

    return df


def student_vs_AI() -> pd.DataFrame:
    path = DATA_FOLDER + "student_vs_AI.csv"

    df = pd.read_csv(path)

    # rename columns
    df = df.rename(columns={
        "Text": "text",
        "Label": "isGenerated"
    })

    # convert labels
    df["isGenerated"] = df["isGenerated"].map({
        "student": 0,
        "ai": 1
    })

    # optional: drop rows that failed mapping
    df = df.dropna(subset=["isGenerated"])

    df.to_csv("data/student_vs_AI_cleaned.csv", index=False)

    return df

def Training_Essay_Dataset() -> pd.DataFrame:
    path = DATA_FOLDER + "Training_Essay_Data.csv"

    df = pd.read_csv(path)

    # rename columns
    df = df.rename(columns={
        "text": "text",
        "generated": "isGenerated"
    })

    # ensure correct type (0/1 integers)
    df["isGenerated"] = df["isGenerated"].astype(int)

    out_path = DATA_FOLDER + "Training_Essay_Data_cleaned.csv"
    df.to_csv(out_path, index=False)

    return df

def AI_Generated_Essays() -> pd.DataFrame:
    path = DATA_FOLDER + "AI Generated Essays Dataset.csv"

    df = pd.read_csv(path)

    # rename columns to your standard schema
    df = df.rename(columns={
        "text": "text",
        "generated": "isGenerated"
    })

    # ensure integer labels
    df["isGenerated"] = df["isGenerated"].astype(int)

    out_path = DATA_FOLDER + "AI_Generated_Essays_cleaned.csv"
    df.to_csv(out_path, index=False)

    return df

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


"""def students_vs_ai() -> pd.DataFrame:
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
"""


AI_Human()
"""ai_vs_human_comparison_dataset()"""
student_vs_AI()
"""Training_Essay_Dataset()"""

"""AI_Generated_Essays() """
