from flask import Flask, request, render_template, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import speech_recognition as sr
import os

app = Flask(__name__)

# Define the upload folder for temporary audio files
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'temp')

# Load fine-tuned model
model_save_path = "fine_tuning/fine_tuned_model"
model = GPT2LMHeadModel.from_pretrained(model_save_path)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define conversation history storage
conversation_history = []

@app.route("/")
def home():
    return render_template("index.html", conversation_history=conversation_history)

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

    return render_template("conversation_history.html", conversation_history=conversation_history)

@app.route("/process_audio", methods=["POST"])
def process_audio():
    # Check if audio data is provided in the request
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio data provided'}), 400

    audio_data = request.files['audio_data']
    
    # Check file extension
    if not audio_data.filename.endswith(('.wav', '.aiff', '.flac')):
        return jsonify({'error': 'Unsupported file format. Please provide a WAV, AIFF, or FLAC file.'}), 400

    # Save audio file to a temporary location
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'recording.wav')
    audio_data.save(file_path)

    recognizer = sr.Recognizer()

    # Use try-except to handle potential errors during recognition
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return jsonify({'text': text}), 200
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand the audio'}), 400
    except sr.RequestError:
        return jsonify({'error': 'Speech recognition service error'}), 500
    finally:
        # Remove the temporary audio file
        os.remove(file_path)

if __name__ == "__main__":
    app.run(debug=True)
