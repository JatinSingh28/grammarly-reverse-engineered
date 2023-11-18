import numpy as np
from dataPreprocessorWordLevel import input_token_index, target_token_index, pad_tok,incorrect_sentences,correct_sentences,max_encoder_seq_len,max_decoder_seq_len,num_decoder_tokens

def init_model(input,max_encoder_seq_len,max_decoder_seq_len,num_decoder_tokens):
    encoder_input_data = np.zeros((len(input),max_encoder_seq_len),dtype='float32')
    decoder_input_data = np.zeros((len(input),max_decoder_seq_len),dtype='float32')
    decoder_target_data = np.zeros((len(input),max_decoder_seq_len,num_decoder_tokens),dtype='float32')
    
    return encoder_input_data,decoder_input_data,decoder_target_data

def generate_data(input,target,max_encoder_seq_len, max_decoder_seq_len, num_decoder_tokens):
    encoder_input_data, decoder_input_data,decoder_target_data = init_model(input,max_encoder_seq_len,max_decoder_seq_len,num_decoder_tokens)
    for i,(input_text, target_text) in enumerate(zip(input,target)):
        for t,tok in enumerate(input_text.split()):
            encoder_input_data[i,t] = input_token_index[tok]
        encoder_input_data[i,t+1:] =input_token_index[pad_tok]
        
        for t, tok in enumerate(target_text.split()):
          # decoder_target_data is ahead of decoder_input_data by one timestep
          decoder_input_data[i, t] = target_token_index[tok]
          if t > 0:
              # decoder_target_data will be ahead by one timestep
              # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[tok]] = 1.
            decoder_input_data[i, t+1:] = target_token_index[pad_tok]
            decoder_target_data[i, t:, target_token_index[pad_tok]] = 1.
    return encoder_input_data, decoder_input_data,decoder_target_data

encoder_input_data, decoder_input_data ,decoder_target_data = generate_data(incorrect_sentences,correct_sentences,max_encoder_seq_len,max_decoder_seq_len,num_decoder_tokens)