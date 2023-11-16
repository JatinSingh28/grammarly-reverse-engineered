# Assuming you have already trained and have the 'model' variable as the trained Seq2Seq model
import numpy as np
import tensorflow as tf
import pickle

# Define a function to correct a single sentence
def correct_sentence(input_sentence):
    dataDict = pickle.load("dataDict")
    model_path = "./models/baseModel/saved_model.pb"
    model = tf.saved_model.load(model_path)
    encoder_input_data = np.zeros((1, dataDict.max_encoder_seq_length, len(dataDict.input_characters)), dtype='float32')
    for t, char in enumerate(input_sentence):
        encoder_input_data[0, t, dataDict.input_token_index[char]] = 1.0

    decoder_input_data = np.zeros((1, max_decoder_seq_length, len(target_characters)), dtype='float32')
    decoder_input_data[0, 0, target_token_index['\t']] = 1.0  # '\t' is the start token

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens = model.predict([encoder_input_data, decoder_input_data])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = [k for k, v in target_token_index.items() if v == sampled_token_index][0]

        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        decoder_input_data = np.zeros((1, max_decoder_seq_length, len(target_characters)), dtype='float32')
        decoder_input_data[0, 0, sampled_token_index] = 1.0

    return decoded_sentence

# Example sentence for correction
input_sentence = 'He hav a red car.'

# Get the corrected sentence
corrected_sentence = correct_sentence(input_sentence)

print("Input Sentence:", input_sentence)
print("Corrected Sentence:", corrected_sentence)
