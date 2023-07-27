# Language Model From Scratch

### This is a well-documented language model that I've built from scratch, defaulting to about 11 million parameters that I have been using to learn about language models. 

This project contains two language models, each broken up into their distinct parts.

- Bigram Language Model: A bigram model is a type of n-gram model that predicts the probability of a word based on the previous word. It considers pairs of consecutive words (bigrams) and calculates the conditional probability of the current word given the previous word.

- GPT Language Model: GPT is a transformer-based language model that uses a deep neural network architecture, specifically the Transformer model. It is a much more sophisticated model that considers the entire context of a sentence, not just the previous word. GPT is based on a self-attention mechanism that allows it to weigh the importance of each word in the context and capture long-range dependencies in text.


## Usage
**Optional:** `python -m venv env`

### Training data
Add your input data to the `data/` folder.
Ensure to update the path to the name of your file in the INPUT_FILE variable if it has a custom name or name it `input.txt`

### Download packages
`pip install -r requirements.txt`

### Run Bigram Model
`python -m models.bigram.train.py`

### Run GPT Model
`python -m models.gpt.train.py`
