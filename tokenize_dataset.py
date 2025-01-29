from transformers import AutoTokenizer
from load_dataset import load_and_preprocess_data

def tokenize_data(dataset):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

if __name__ == "__main__":
    dataset = load_and_preprocess_data()
    tokenized_dataset = tokenize_data(dataset)
    print("Dataset tokenized successfully!")