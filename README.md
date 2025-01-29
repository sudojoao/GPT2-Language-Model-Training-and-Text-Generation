# GPT2-Language-Model-Training-and-Text-Generation
This project demonstrates how to train a GPT-2 language model on the Wikitext-2 dataset and generate text using the trained model.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/gpt2-text-generation.git
    cd gpt2-text-generation
    ```

2. Install the required packages:
    ```sh
    pip install transformers datasets
    ```

3. Install pytorch:
   ```sh
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Training the Model

To train the model, run the [train_model.py](http://_vscodecontentref_/0) script. This will load the Wikitext-2 dataset, tokenize it, and train a GPT-2 model. The model and tokenizer will be saved in the `results/latest_checkpoint` directory.

```sh
python train_model.py
```
You can either allow the model to train until completion for optimal accuracy, or you can interrupt the training process at any time, which may result in reduced accuracy.

## Generating Text on User Input

To generate text using the trained model, run the `generate_text.py` script. This will load the model and tokenizer from the `results/latest_checkpoint` directory and generate text based on the input prompt.

```sh
python generate_text.py
```
