import pandas as pd
import numpy as np
import pickle
import string
import re

# from tensorflow.keras.preprocessing.text import Tokenizer

n_rows = 1000
df = pd.read_csv("../data/data1.csv", nrows=n_rows)

incorrect_sentences = df["input"]
correct_sentences = df["target"]


def clean(sentences):
    sentences = sentences.apply(lambda x: x.lower())

    # To help the model capture the word separations, mark the comma with special token:
    sentences = sentences.apply(lambda x: re.sub(",", " COMMA", x))

    exclude_punctuations = str.maketrans("", "", string.punctuation)
    sentences = sentences.apply(lambda x: x.translate(exclude_punctuations))

    exclude_digits = str.maketrans("", "", string.digits)
    sentences = sentences.apply(lambda x: x.translate(exclude_digits))
    return sentences


def addToken(sentences):
    st_tok = "START_"
    end_tok = "_END"

    sentences = clean(sentences)
    sentences = sentences.apply(lambda x: st_tok + " " + x + " " + end_tok)
    return sentences


def split(data):
    return data.split()


def dataStats(inputs, targets):
    def tokenSplit(data):
        return data.split()

    inputTokens = set()
    for i in inputs:
        for tok in tokenSplit(i):
            inputTokens.add(tok)
    targetTokens = set()
    for i in targets:
        for tok in tokenSplit(i):
            targetTokens.add(tok)

    inputTokens = sorted(list(inputTokens))
    targetTokens = sorted(list(targetTokens))

    num_encoder_tokens = len(inputTokens)
    num_decoder_tokens = len(targetTokens)
    max_encoder_seq_len = np.max([len(tokenSplit(i)) for i in inputs])
    max_decoder_seq_len = np.max([len(tokenSplit(i)) for i in targets])

    print("Number of samples:", len(inputs))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_len)
    print("Max sequence length for outputs:", max_decoder_seq_len)

    return (
        inputTokens,
        targetTokens,
        num_encoder_tokens,
        num_decoder_tokens,
        max_encoder_seq_len,
        max_decoder_seq_len,
    )


def vocab(inputTokens, targetTokens):
    input_token_index = {}
    target_token_index = {}
    for i, tok in enumerate(special_tokens):
        input_token_index[tok] = i
        target_token_index[tok] = i

    offset = len(special_tokens)
    for i, tok in enumerate(inputTokens):
        input_token_index[tok] = i + offset
    for i, tok in enumerate(targetTokens):
        target_token_index[tok] = i + offset

    # Reverse lookup token index to decode sequence back

    reverse_input_tok_index = dict((i, tok) for tok, i in input_token_index.items())
    reverse_target_tok_index = dict((i, tok) for tok, i in target_token_index.items())

    return (
        input_token_index,
        target_token_index,
        reverse_input_tok_index,
        reverse_target_tok_index,
    )


# if __name__ == "__main__":
    
inputTokens,targetTokens,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_len,max_decoder_seq_len,= dataStats(incorrect_sentences, correct_sentences)

pad_tok = "PAD"
sep_tok = " "
st_tok = "_START"
end_tok = "END_"
special_tokens = [pad_tok, sep_tok, st_tok, end_tok]
num_encoder_tokens += len(special_tokens)
num_decoder_tokens += len(special_tokens)

max_encoder_seq_length = 500
max_decoder_seq_length = 250

(
    input_token_index,
    target_token_index,
    reverse_input_tok_index,
    reverse_target_tok_index,
) = vocab(inputTokens, targetTokens)
