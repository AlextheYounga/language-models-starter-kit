import torch
from .bigram_language_model import BigramLanguageModel
from models.encoder import Encoder
from .hyperparams import *


INPUT_FILE = 'data/input.txt'
TEXT = open(INPUT_FILE, 'r', encoding='utf-8').read()
MODEL_NAME = 'alex_tweets_model'
CHECKPOINT_DIRECTORY = 'storage/checkpoints'


def create_datasets(data):
    # Let's now split up the data into train and validation sets
    n = int(0.9*len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def get_batch(split):
    """
    Generate a small batch of data of inputs x and targets y

    Explanation:
    The block size represents chunks of predictions, where given 18, 47 comes next, and so on.
    Although there are 9 characters total here, there are 8 individual predictions.
    => tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])

    Example:
    for t in range(block_size):
        context = x[:t+1]
        target = y[t]
        print(f"when input is {context} the target: {target}")
    """

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ==== Run Training ==== #

# Encode text -> see encoder.py for more info.
encoder = Encoder(TEXT)
vocab_size = encoder.vocab_size  # Number of characters in text

# Pass encoded text into PyTorch tensor
data = torch.tensor(encoder.encode(TEXT), dtype=torch.long)
train_data, val_data = create_datasets(data)


if __name__ == "__main__":
    model = BigramLanguageModel(vocab_size)
    m = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output = encoder.decode(m.generate(context, max_new_tokens=500)[0].tolist())
    print(output)
