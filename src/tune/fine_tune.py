from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from .hyperparameters import *


INPUT_FILE = 'data/input.txt'
TEXT = open(INPUT_FILE, 'r', encoding='utf-8').read()
PRETRAINED_MODEL = '/home/alexyounger/Documents/Develop/AI/Testing/text-generation-webui/models/gpt2'
TUNED_MODEL_SAVE_DIRECTORY = 'storage/trained_models/alex_model_gpt2'


class FineTune():
    def load_dataset(self, file_path, tokenizer):
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=block_size,
        )
        return dataset

    def load_data_collator(self, tokenizer):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
        )
        return data_collator

    def train(self):
        tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_MODEL)

        train_dataset = self.load_dataset(INPUT_FILE, tokenizer)

        data_collator = self.load_data_collator(tokenizer)

        tokenizer.save_pretrained(TUNED_MODEL_SAVE_DIRECTORY)

        model = GPT2LMHeadModel.from_pretrained(PRETRAINED_MODEL)

        model.save_pretrained(TUNED_MODEL_SAVE_DIRECTORY)

        training_args = TrainingArguments(
            output_dir=TUNED_MODEL_SAVE_DIRECTORY,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        trainer.train()
        trainer.save_model()


# Train
tune = FineTune()

tune.train()
