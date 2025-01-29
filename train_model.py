from transformers import AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from tokenize_dataset import tokenize_data
from load_dataset import load_and_preprocess_data

def train_model(tokenized_dataset):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=512,
        n_embd=256,
        n_layer=6,
        n_head=8,
    )
    
    model = AutoModelForCausalLM.from_config(config)
    
    training_args = TrainingArguments(
        output_dir="./results/latest_checkpoint",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=100,  
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,
    )
    
    tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model...")
        model.save_pretrained("./results/latest_checkpoint")
        tokenizer.save_pretrained("./results/latest_checkpoint")
        print("Model saved. Exiting...")
        exit(0)

    model.save_pretrained("./results/latest_checkpoint")
    tokenizer.save_pretrained("./results/latest_checkpoint")

if __name__ == "__main__":
    dataset = load_and_preprocess_data()
    tokenized_dataset = tokenize_data(dataset)
    train_model(tokenized_dataset)