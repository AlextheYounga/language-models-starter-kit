# Inference
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys

"""
For a better chat experience, I recommend this repository:
https://github.com/oobabooga/text-generation-webui
"""

MODEL = 'path/to/your/model'
MAX_LENGTH = 50

class ChatBot():
    def load_model(self):
        model = GPT2LMHeadModel.from_pretrained(MODEL)
        return model

    def load_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
        return tokenizer
    
    def model_response(self, model, tokenizer, sequence):
        ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=MAX_LENGTH,
            pad_token_id=model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )

        response = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
        return response
    
    def run(self):
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        print('Chat started...\n')

        try:
            while True:
                sequence = input()

                if sequence:
                    output = self.model_response(model, tokenizer, sequence)
                    sys.stdout.write(output + "\n\n")
                else:
                    sys.stdout.write("Please enter your prompt:")
        except KeyboardInterrupt:
            print('Shutting down...')
            sys.exit()

chat = ChatBot()
chat.run()