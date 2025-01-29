from datasets import load_dataset

def load_and_preprocess_data():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    return dataset