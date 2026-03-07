import json 
import pandas as pd


# Fonksiyonlar -> veriyi alıp temizleyip veri_clean.csv haline kaydeder
# temizlenen csv format: text, isGenerated (0/1)     Human = 0, AI = 1

DATA_FOLDER = "data/"
CLEANED_DATA_FOLDER = "data/cleaned/"

def _save_to_disk(df: pd.DataFrame, filename: str): # removes duplicates, deletes null values and saves with filename
    df.drop_duplicates(subset=["text", "isGenerated"], inplace=True)
    df.dropna(subset=["text", "isGenerated"], inplace=True)

    path_to_save = CLEANED_DATA_FOLDER +"cleaned_" +filename
    df.to_csv(path_to_save, index=False)


def AI_Human() :
    path = "data/AI_Human.csv"
    df = pd.read_csv(path)
    # df["generated"] = 1 - df["generated"] 
    df = df.rename(columns={
        "Text": "text", 
        "generated": "isGenerated"
    })
    # df.to_csv("data/AI_Human_cleaned.csv", index=False)
    _save_to_disk(df, "AI_Human_cleaned.csv")

def ai_vs_human_comparison_dataset():
    path = "data/ai-vs-human-comparison-dataset.csv"
    df = pd.read_csv(path)

    df = df[["text", "label"]]

    labels = df["label"].astype(str).str.strip().str.lower()

    mapping = {"human": 0, "ai": 1}
    df["label"] = labels.map(mapping)

    df = df.rename(columns={"label": "isGenerated"})
    df["isGenerated"] = df["isGenerated"].astype(int)
 
    _save_to_disk(df, "ai_vs_human_comparison_dataset_cleaned.csv")
 


def student_vs_AI() :
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

    df["isGenerated"] = df["isGenerated"].astype(int)
    _save_to_disk(df, "student_vs_AI_cleaned.csv")

def Training_Essay_Dataset() :
    path = DATA_FOLDER + "Training_Essay_Data.csv"

    df = pd.read_csv(path)

    # rename columns
    df = df.rename(columns={
        "text": "text",
        "generated": "isGenerated"
    })

    # ensure correct type (0/1 integers)
    df["isGenerated"] = df["isGenerated"].astype(int)
    _save_to_disk(df, "Training_Essay_Data_cleaned.csv")
 
def AI_Generated_Essays() :
    path = DATA_FOLDER + "AI Generated Essays Dataset.csv"

    df = pd.read_csv(path)

    # rename columns to your standard schema
    df = df.rename(columns={
        "text": "text",
        "generated": "isGenerated"
    })

    # ensure integer labels
    df["isGenerated"] = df["isGenerated"].astype(int)

    _save_to_disk(df, "AI_Generated_Essays_cleaned.csv")

def humanVSAIJSONL() -> list: # DEPRECATED, we don't use jsonl format
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

    _save_to_disk(df, "humanVSAIJSONL_cleaned.csv")


"""def students_vs_ai() :
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


"""AI_Human()
ai_vs_human_comparison_dataset()
student_vs_AI()
Training_Essay_Dataset()"""

AI_Generated_Essays() 
# AI_Human()