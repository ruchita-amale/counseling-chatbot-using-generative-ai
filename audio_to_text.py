import os
import whisper
vall=1
# Create output directory if it doesn't exist
output_dir = "Data/Output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load model
model = whisper.load_model("base")

# Process each file in the input directory
input_dir = "Data/Input/"
for filename in os.listdir(input_dir):
    if filename.endswith(".mp3"):  # Process only .mp3 files
        input_file_path = os.path.join(input_dir, filename)

        # Extract base name of input file (without extension)
        file_name_base = os.path.splitext(filename)[0]

        output_file_path = os.path.join(output_dir, file_name_base + ".txt")

        # Transcribe audio
        result = model.transcribe(input_file_path)

        # Write result to output file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(result["text"])
        print(vall)
    elif filename.endswith(".pdf"):  # Process PDF files
        input_file_path = os.path.join(input_dir, filename)
        file_name_base = os.path.splitext(filename)[0]
        output_file_path = os.path.join(output_dir, file_name_base + ".txt")

        # Convert PDF to text
        with open(input_file_path, "rb") as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ""
            for page in range(reader.numPages):
                text += reader.getPage(page).extractText()

        # Write result to output file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(text)
        print("Converted", filename, "to text.")

    elif filename.endswith((".mp4", ".avi", ".mov")):  # Process video files
        input_file_path = os.path.join(input_dir, filename)
        file_name_base = os.path.splitext(filename)[0]
        output_file_path = os.path.join(output_dir, file_name_base + ".txt")

        # Convert video to audio
        audio_file_path = os.path.join(output_dir, file_name_base + ".mp3")
        clip = mp.VideoFileClip(input_file_path)
        clip.audio.write_audiofile(audio_file_path)

        # Transcribe audio
        result = model.transcribe(audio_file_path)

        # Write result to output file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(result["text"])
        print("Converted", filename, "to text.")

    else:
        print("Unsupported file format:", filename)
    val=vall+1