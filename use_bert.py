import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned model and tokenizer
model_path = "./BERT"  # Path to the directory containing the saved model
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)


# Function to perform sentiment analysis
def analyze_sentiment(prompt):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Forward pass through the model
    outputs = model(**inputs)

    # Get the predicted class (0 for negative, 1 for positive)
    predicted_class = torch.argmax(outputs.logits).item()

    # Map predicted class to sentiment label
    sentiment_label = "positive" if predicted_class == 1 else "negative"

    return sentiment_label


# Example usage
user_prompt = "I enjoyed the movie, it was great!"
sentiment = analyze_sentiment(user_prompt)
print("Sentiment:", sentiment)
