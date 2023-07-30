# Inference
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from .hyperparameters import max_length


MODEL='/home/alexyounger/Documents/Develop/AI/Testing/text-generation-webui/models/gpt2'

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



sequence = "Tell me about"

infer = Inference()

infer.generate_text(sequence)