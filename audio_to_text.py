import os
import whisper  # Library for audio transcription
import PyPDF2  # Library for PDF processing
from mpmath import mp  # Library for video to audio conversion

# Create output directory if it doesn't exist
output_dir = "Data/Output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load model for audio transcription
model = whisper.load_model("base")

# Process each file in the input directory
input_dir = "Data/Input/"
for filename in os.listdir(input_dir):
    # Process audio files (MP3)
    if filename.endswith(".mp3"):
        input_file_path = os.path.join(input_dir, filename)

        # Extract base name of input file (without extension)
        file_name_base = os.path.splitext(filename)[0]

        # Define output file path with .txt extension
        output_file_path = os.path.join(output_dir, file_name_base + ".txt")

        # Transcribe audio using the loaded model
        result = model.transcribe(input_file_path, language="en")

        # Write transcribed text to output file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(result["text"])
        print(filename)

    # Process PDF files
    elif filename.endswith(".pdf"):
        input_file_path = os.path.join(input_dir, filename)
        file_name_base = os.path.splitext(filename)[0]

        # Define output file path with .txt extension
        output_file_path = os.path.join(output_dir, file_name_base + ".txt")

        # Extract text from PDF file
        with open(input_file_path, "rb") as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ""
            for page in range(reader.numPages):
                text += reader.getPage(page).extractText()

        # Write extracted text to output file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(text)
        print("Converted", filename, "to text.")

    # Process video files (MP4, AVI, MOV)
    elif filename.endswith((".mp4", ".avi", ".mov")):
        input_file_path = os.path.join(input_dir, filename)
        file_name_base = os.path.splitext(filename)[0]

        # Define output file path with .txt extension
        output_file_path = os.path.join(output_dir, file_name_base + ".txt")

        # Convert video to audio
        audio_file_path = os.path.join(output_dir, file_name_base + ".mp3")
        clip = mp.VideoFileClip(input_file_path)
        clip.audio.write_audiofile(audio_file_path)

        # Transcribe audio using the loaded model
        result = model.transcribe(audio_file_path)

        # Write transcribed text to output file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(result["text"])
        print("Converted", filename, "to text.")

    else:
        print("Unsupported file format:", filename)
