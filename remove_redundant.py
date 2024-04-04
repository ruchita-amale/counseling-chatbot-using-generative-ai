# Function to remove the redundant sentence at the end of the file
def remove_redundant_sentence(file_path):
    # Read the text from the file
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Split the text into sentences
    sentences = text.split('.')

    # Get the last sentence
    last_sentence = sentences[-1].strip()

    # Check if the last sentence is repeated at the end of the file
    if len(sentences) > 1 and sentences[-2].strip() == last_sentence:
        # Remove all occurrences of the last sentence except the last one
        occurrences = text.count(last_sentence)
        text = text.rsplit(last_sentence, occurrences - 1)[0]

        # Write the modified text back to the file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)


# Example usage:
file_path = "Data/Output/hello.txt"
remove_redundant_sentence(file_path)