import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention
import tensorflow_addons as tfa
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from dataPreprocessor import n_rows, input_characters, target_characters, encoder_input_data, decoder_input_data, decoder_target_data

#configrations
epochs = 1
batch_size = 64
validation_split = 0.2
model_name = "seq2seq + Attention"

wandb.init(
    # set the wandb project where this run will be logged
    project="grammarly-reverse-engineered-base-model",

    # track hyperparameters and run metadata with wandb.config
    config={
        "data:":"C4 200M",
        "model": model_name,
        "files_taken":1,
        "n_rows":n_rows,
        "optimizer":"adam", 
        "loss":"categorical_crossentropy", 
        "metric": "accuracy",
        "latent_dim":256,
        "epoch": epochs,
        "batch_size": batch_size,
        "validation_split":validation_split
    }
)

config = wandb.config

# Define Seq2Seq model with Attention
latent_dim = 256  # dimensionality of the encoding space

encoder_inputs = Input(shape=(None, len(input_characters)))
encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, len(target_characters)))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Attention mechanism
attention_layer = Attention()
context_vector = attention_layer([decoder_outputs, encoder_outputs])
decoder_combined_context = tf.concat([context_vector, decoder_outputs], axis=-1)

# Dense layer to generate output
decoder_dense = Dense(len(target_characters), activation='softmax')
output = decoder_dense(decoder_combined_context)

# Define the model
model = Model([encoder_inputs, decoder_inputs], output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
tqdm_callback = tfa.callbacks.TQDMProgressBar()
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
          callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models"),tqdm_callback])

model.save("models/"+model_name)
wandb.finish()
