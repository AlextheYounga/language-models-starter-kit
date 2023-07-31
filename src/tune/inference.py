# Inference
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from .hyperparameters import max_length
import sys


MODEL = '/path/to/your/pretrained/model/directory'


class Inference():
    def load_model(self):
        model = GPT2LMHeadModel.from_pretrained(MODEL)
        return model

    def load_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
        return tokenizer

    def generate_text(self, sequence):
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )
        print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

    def chatbot(self):
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        print('Chat started...\n')

        try:
            while True:
                sequence = input()

                if sequence:
                    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
                    final_outputs = model.generate(
                        ids,
                        do_sample=True,
                        max_length=max_length,
                        pad_token_id=model.config.eos_token_id,
                        top_k=50,
                        top_p=0.95,
                    )
                    sys.stdout.write(tokenizer.decode(final_outputs[0], skip_special_tokens=True) + "\n\n")
                else:
                    sys.stdout.write("Please enter your prompt:")
        except KeyboardInterrupt:
            print('Shutting down...')
            sys.exit()


# sequence = "Tell me about"
# infer.generate_text(sequence)
infer = Inference()
infer.chatbot()
