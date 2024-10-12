import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


def preprocess_video_data(input_file, output_file, max_transcript_length=500, max_title_length=20, num_samples=100000,
                          vocab_size=20000):
    # Load the raw data
    data = pd.read_csv(input_file)

    # Limit the number of samples
    if num_samples < len(data):
        data = data.sample(n=num_samples, random_state=42)

    # Clean and preprocess transcripts and titles
    data['transcript'] = data['transcript'].apply(clean_text)
    data['title'] = data['title'].apply(clean_text)

    # Add start and end tokens to titles
    data['title'] = 'startseq ' + data['title'] + ' endseq'

    # Create and fit the tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['transcript'].tolist() + data['title'].tolist())

    # Tokenize transcripts and titles
    transcript_sequences = tokenizer.texts_to_sequences(data['transcript'])
    title_sequences = tokenizer.texts_to_sequences(data['title'])

    # Pad sequences
    transcript_padded = pad_sequences(transcript_sequences, maxlen=max_transcript_length, padding='post',
                                      truncating='post')
    title_padded = pad_sequences(title_sequences, maxlen=max_title_length, padding='post', truncating='post')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(transcript_padded, title_padded, test_size=0.2, random_state=42)

    # Save preprocessed data
    np.savez(output_file,
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)

    # Save tokenizer using pickle
    with open(output_file + "_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"Preprocessed data saved to {output_file}")
    print(f"Tokenizer saved to {output_file}_tokenizer.pkl")
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")


if __name__ == "__main__":
    preprocess_video_data("YT-titles-transcripts-clean.csv", "preprocessed_video_data.npz")