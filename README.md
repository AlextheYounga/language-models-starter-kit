# Language Models Starter Kit

## This is everything you need to get started building your own GPT models from the ground up, or fine-tuning pre-existing models using your own data. 

This project contains two main directories, `models` and `tune`:
```
├── src
│   ├── models
│   │   ├── bigram
│   │   ├── encoder.py
│   │   └── gpt
│   └── tune
│       ├── chatbot.py
│       ├── fine_tune.py
│       ├── hyperparameters.py
│       └── inference.py
```
### Model Concepts
You can play around with the concepts in language models in the `models` folder. Here there are two kinds of LM strategies,
each broken up into their distinct parts:

- **Bigram Language Model**: A bigram model is a type of n-gram model that predicts the probability of a word based on the previous word. It considers pairs of consecutive words (bigrams) and calculates the conditional probability of the current word given the previous word.

- **GPT Language Model**: GPT is a transformer-based language model that uses a deep neural network architecture, specifically the Transformer model. It is a much more sophisticated model that considers the entire context of a sentence, not just the previous word. GPT is based on a self-attention mechanism that allows it to weigh the importance of each word in the context and capture long-range dependencies in text.

### Fine Tuning
You can fine-tune pre-existing models using your own training data. This is fairly straightforward, and all the primary logic for that takes place inside the `src/tune/fine_tune.py` file. You can place your existing model anywhere as long as you specify the path to that model's directory with the `PRETRAINED_MODEL` constant inside the `fine_tune.py` file.

### Inference
I have two methods of testing inference with the model built into the `src/tune` folder. You can run the `src/tune/inference.py` file, or test it out in a simple terminal chat interface in the `src/tune/chatbot.py` file.

### Usage
*Optional: Create an env* 
```bash
python -m venv env

source env/bin/activate
```

##### Training data
Add your input data to the `data/` folder.
Ensure to update the path to the name of your file in the INPUT_FILE variable if it has a custom name or name it `input.txt`

##### Download packages
```bash
pip install -r requirements.txt
```


### Building your own language model from the ground up

This project contains two language models, 

#### Run Bigram Model
```bash
python -m src.models.bigram.train
```

#### Run GPT Model
```bash
python -m src.models.gpt.train
```


### Fine Tuning Existing Models on Your Own Data

Ensure you have specified the model path inside the file. You can place your model anywhere as long as you specify the path to its directory inside of the `fine_tune.py` file.

#### Run Fine Tuning on PreExisting Model
```bash
python -m src.tune.fine_tune
```

##### Test Inference
```bash
python -m src.tune.inference
```

##### Run ChatBot
```bash
python -m src.tune.chatbot
```
