import os
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Define paths to input files and output folder
input_dir = "Data/Output"
model_dir = "Model"

# Check if the output directory exists, if not, create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Load data from JSON files
qa_data = []
for file_name in os.listdir(input_dir):
    if file_name.endswith('.json'):
        with open(os.path.join(input_dir, file_name), 'r') as f:
            qa_data.extend(json.load(f))

# Load data from TXT files
txt_data = []
for file_name in os.listdir(input_dir):
    if file_name.endswith('.txt'):
        with open(os.path.join(input_dir, file_name), 'r', encoding='utf-8') as f:
            txt_data.append(f.read())

# Combine data
combined_data = qa_data + txt_data

# Initialize GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize the data
tokenized_data = tokenizer(combined_data, truncation=True, padding=True)

class MyDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.tokenized_data.items()}

# Create dataset and dataloader
dataset = MyDataset(tokenized_data)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator)

# Define training arguments
training_args = TrainingArguments(
    output_dir=model_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print("Fine-tuning and saving complete!")
