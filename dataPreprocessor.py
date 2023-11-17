import pandas as pd
import numpy as np
import pickle


n_rows = 10
df = pd.read_csv("./data/data1.csv", nrows=n_rows)

incorrect_sentences = df["input"]
correct_sentences = df["target"]

# Generate vocabulary sets
input_characters = set(" ".join(incorrect_sentences + correct_sentences))
target_characters = input_characters

# Generate character-level index mappings
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = input_token_index

# Define max sequence lengths
max_encoder_seq_length = max([len(txt) for txt in incorrect_sentences])
max_decoder_seq_length = max([len(txt) for txt in correct_sentences])

dataDict = {
    "input_characters": input_characters,
    "target_characters": target_characters,
    "input_token_index": input_token_index,
    "target_token_index": target_token_index,
    "max_encoder_seq_length": max_encoder_seq_length,
    "max_decoder_seq_length": max_decoder_seq_length,
}

file_path = "dataDict"
with open(file_path, "wb") as file:
    pickle.dump(dataDict, file)

# Create encoder and decoder data
encoder_input_data = np.zeros(
    (len(incorrect_sentences), max_encoder_seq_length, len(input_characters)),
    dtype="float32",
)
decoder_input_data = np.zeros(
    (len(incorrect_sentences), max_decoder_seq_length, len(target_characters)),
    dtype="float32",
)
decoder_target_data = np.zeros(
    (len(incorrect_sentences), max_decoder_seq_length, len(target_characters)),
    dtype="float32",
)

for i, (input_text, target_text) in enumerate(
    zip(incorrect_sentences, correct_sentences)
):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
