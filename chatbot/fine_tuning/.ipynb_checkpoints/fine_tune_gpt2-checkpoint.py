import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Define paths
json_dir = "data/json_file"
text_dir = "data/text_files"
preprocessed_output_dir = "data/preprocessed_output"
model_save_path = "fine_tuned_model"

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-small")

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokenized_text = tokenizer.tokenize(text)

    # Replace unknown words with <unk>
    tokenized_text = [token if token in tokenizer.vocab else tokenizer.unk_token for token in tokenized_text]

    # Remove special characters and symbols
    cleaned_text = [token for token in tokenized_text if token.isalnum() or token in ["!", ".", "?"]]

    # Convert tokens to lowercase
    cleaned_text_lower = [token.lower() for token in cleaned_text]

    # Join tokens into a single string
    preprocessed_text = " ".join(cleaned_text_lower)
    
    return preprocessed_text

# Function to preprocess JSON files
def preprocess_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        questions_answers = data.get("questions_answers", [])
        preprocessed_texts = []
        for qa in questions_answers:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            preprocessed_question = preprocess_text(question)
            preprocessed_answer = preprocess_text(answer)
            preprocessed_text = f"{preprocessed_question} {preprocessed_answer}"
            preprocessed_texts.append(preprocessed_text)
        return preprocessed_texts

# Function to preprocess text files
def preprocess_text_file(file_path):
    with open(file_path, "r") as f:
        text = f.read()
        preprocessed_text = preprocess_text(text)
        return preprocessed_text

# Preprocess JSON files
preprocessed_json_files = []
for file_name in os.listdir(json_dir):
    if file_name.endswith(".json"):
        file_path = os.path.join(json_dir, file_name)
        preprocessed_texts = preprocess_json_file(file_path)
        preprocessed_json_files.extend(preprocessed_texts)

# Preprocess text files
preprocessed_text_files = []
for file_name in os.listdir(text_dir):
    if file_name.endswith(".txt"):  # Assuming all files are text files
        file_path = os.path.join(text_dir, file_name)
        preprocessed_text = preprocess_text_file(file_path)
        preprocessed_text_files.append(preprocessed_text)

# Create preprocessed output directory if it doesn't exist
if not os.path.exists(preprocessed_output_dir):
    os.makedirs(preprocessed_output_dir)

# Save preprocessed text files
for i, preprocessed_text in enumerate(preprocessed_json_files + preprocessed_text_files):
    output_file_path = os.path.join(preprocessed_output_dir, f"preprocessed_{i}.txt")
    with open(output_file_path, "w") as f:
        f.write(preprocessed_text)

# Create dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    text_files=[os.path.join(preprocessed_output_dir, file_name) for file_name in os.listdir(preprocessed_output_dir)],
    block_size=128
)

# Initialize model
model = GPT2LMHeadModel.from_pretrained("gpt2-small")

# Define training arguments
training_args = TrainingArguments(
    output_dir=model_save_path,
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
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
    train_dataset=dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(model_save_path)