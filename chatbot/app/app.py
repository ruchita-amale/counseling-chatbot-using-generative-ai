from flask import Flask, request, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load fine-tuned model
model_save_path = "fine_tuned_model"
model = GPT2LMHeadModel.from_pretrained(model_save_path)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-small")

# Define conversation history storage
conversation_history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    
    # Append user input to conversation history
    conversation_history.append({"user": user_input})

    # Generate response
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    response = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)
    bot_response = tokenizer.decode(response[0], skip_special_tokens=True)

    # Append bot response to conversation history
    conversation_history[-1]["bot"] = bot_response

    return render_template("index.html", conversation_history=conversation_history)

if __name__ == "__main__":
    app.run(debug=True)
