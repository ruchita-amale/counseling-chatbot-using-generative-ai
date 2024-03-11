import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Set up paths
data_dir = "Data/Output"
output_dir = "fine_tuned_GPT-2"

# Function to read text files and concatenate the text
def read_text_files(directory):
    corpus = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='latin-1') as file:
                corpus += file.read() + "\n"
    return corpus

print("Set up paths")

# Read text from files and concatenate
corpus = read_text_files(data_dir)
print("Read text from files and concatenate")

# Save the concatenated corpus
with open(os.path.join(data_dir, "corpus.txt"), 'w', encoding='latin-1') as corpus_file:
    corpus_file.write(corpus)
print("Save the concatenated corpus")

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Prepare the dataset directly from file
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=os.path.join(data_dir, "corpus.txt"),
    block_size=128
)

print("Prepare the dataset")

# Initialize the model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,  # output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    save_steps=500,  # number of updates steps before checkpoint saves
    save_total_limit=2,  # limit the total amount of checkpoints
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
