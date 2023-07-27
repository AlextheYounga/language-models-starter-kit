import torch
from ..encoder import Encoder
from .hyperparams import *
from .gpt_language_model import GPTLanguageModel


INPUT_FILE = 'data/input.txt'
TEXT = open(INPUT_FILE, 'r', encoding='utf-8').read()
MODEL_NAME = 'alex_tweets_model'
CHECKPOINT_DIRECTORY = 'storage/checkpoints'


def create_datasets(data):
    # Let's split up the data into train and validation sets
    n = int(0.9*len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def save_checkpoint(parameters_number, model, optimizer, iter):
    # Define a dictionary to store necessary information for resuming training or inference
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': iter,
        # Add any other relevant information you might need
    }
    params_number_string = f"{round(parameters_number, 2)}m"
    checkpoint_file_name = f"{MODEL_NAME}_{params_number_string}_gpt_checkpoint.pth"

    # Save the checkpoint to a file
    torch.save(checkpoint, f"{CHECKPOINT_DIRECTORY}/{checkpoint_file_name}")


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


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    tensor = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in tensor])
    y = torch.stack([data[i+1:i+block_size+1] for i in tensor])
    x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    # Setup model
    model = GPTLanguageModel(vocab_size)
    m = model.to(device)

    # print the number of parameters in the model
    parameters_number = sum(p.numel() for p in m.parameters())/1e6
    print(parameters_number, 'M parameters')
    # Default -> 10.898896 M parameters

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Save a checkpoint of our model
            save_checkpoint(parameters_number, model, optimizer, iter)

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    print(encoder.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
