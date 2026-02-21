import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    texts = df["symptoms_text"].astype(str).tolist()
    labels = df["triage_class"].astype(str).tolist()
    return texts, labels