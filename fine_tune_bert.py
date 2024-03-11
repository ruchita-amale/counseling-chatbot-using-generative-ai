import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TextDataset, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments

# Function to read text files and concatenate the text
def read_text_files(directory, encoding='utf-8'):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding=encoding) as file:
                text += file.read() + " "  # Concatenate text from each file
    return text

# Directory containing the text files
text_files_directory = "Data/Input"
output_dir = "fine_tuned_BERT"

# Read text from files
corpus_text = read_text_files(text_files_directory, encoding='latin-1')

# Tokenize the text using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer(corpus_text, truncation=True, padding=True)

# Create a PyTorch dataset
dataset = TextDataset(tokenized_text, tokenized_text)

# Define the model and training arguments
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    output_dir=output_dir,  # Directory to save the fine-tuned model
)

# Create a data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
