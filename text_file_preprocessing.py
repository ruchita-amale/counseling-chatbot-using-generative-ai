def remove_redundant_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split content into sentences
    sentences = content.split('.')

    # Identify repeating sentences at the end
    repeated_sentences = []
    for i in range(len(sentences)-1, 0, -1):
        if sentences[i].strip() == sentences[i-1].strip():
            repeated_sentences.append(sentences[i].strip())
        else:
            break

    # Remove repeating sentences from the end
    if repeated_sentences:
        last_repeated_sentence = repeated_sentences[-1]
        modified_content = content.rsplit(last_repeated_sentence, 1)[0].strip()
    else:
        modified_content = content.strip()

    # Write modified content back to file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)

# Usage example
remove_redundant_sentences("Data/Output/2.4A Blessing Everyone-A.txt")
