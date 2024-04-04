import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load IMDb and Amazon Reviews datasets
imdb_dataset = load_dataset("imdb")
amazon_dataset = load_dataset("amazon_polarity")

# Combine datasets
combined_dataset = imdb_dataset['train']

# Shuffle combined dataset
combined_dataset = combined_dataset.shuffle()

# Tokenize input data
inputs = tokenizer(combined_dataset['text'], padding=True, truncation=True, return_tensors='pt')

# Convert labels to tensors
labels = torch.tensor(combined_dataset['label'])

# Create dataset and data loader
dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# Fine-tune BERT model
num_epochs = 3
model.train()
for epoch in range(num_epochs):
    for batch in loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model_path = "sentiment_model"
torch.save(model.state_dict(), model_path)
print("Model saved at:", model_path)
