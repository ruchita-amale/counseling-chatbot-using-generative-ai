import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
model_dir = "fine_tuned_GPT-2"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input text for generation
input_text = "Your input text here"

# Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate output
output = model.generate(
    input_ids=input_ids,
    max_length=50,  # adjust as needed
    num_return_sequences=1,  # adjust as needed
    temperature=0.7,  # adjust as needed
    pad_token_id=tokenizer.eos_token_id,
)

# Decode the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Store the result in a string
print(output_text)