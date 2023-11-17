import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense
import tensorflow_addons as tfa
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import pickle
from dataPreprocessor import n_rows, input_characters, target_characters, encoder_input_data, decoder_input_data, decoder_target_data

#configrations
epochs = 1
batch_size = 64
validation_split = 0.2
model_name = "seq2seqRNN"

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


# latent_dim = 256  # dimensionality of the encoding space

# train_dataset = tf.data.Dataset.from_tensor_slices(
#     ({"encoder_inputs": encoder_input_data, "decoder_inputs": decoder_input_data}, decoder_target_data)
# ).shuffle(len(encoder_input_data)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# val_size = int(validation_split * len(encoder_input_data))

# # Split the dataset into training and validation
# train_dataset = train_dataset.skip(val_size)
# val_dataset = train_dataset.take(val_size)

# # Encoder
# encoder_inputs = Input(shape=(None, len(input_characters)))
# encoder = SimpleRNN(latent_dim, return_state=True)
# encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# encoder_states = [state_h, state_c]

# # Decoder
# decoder_inputs = Input(shape=(None, len(target_characters)))
# decoder_rnn = SimpleRNN(latent_dim, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_rnn(decoder_inputs, initial_state=encoder_states)
# decoder_dense = Dense(len(target_characters), activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)

# # Define the model
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# tqdm_callback = tfa.callbacks.TQDMProgressBar()
# # model.fit(train_dataset, epochs=epochs, validation_data=val_dataset,
# #           callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models"),tqdm_callback])
          
# # with tf.device('/GPU:0'):
# model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
#           callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models"),tqdm_callback])

# model.save("models/"+model_name)
# wandb.finish()

# Define Seq2Seq model
latent_dim = 256  # dimensionality of the encoding space

train_dataset = tf.data.Dataset.from_tensor_slices(
    ({"encoder_inputs": encoder_input_data, "decoder_inputs": decoder_input_data}, decoder_target_data)
).shuffle(len(encoder_input_data)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_size = int(validation_split * len(encoder_input_data))

# Split the dataset into training and validation
train_dataset = train_dataset.skip(val_size)
val_dataset = train_dataset.take(val_size)

# Encoder
encoder_inputs = Input(shape=(None, len(input_characters)))
encoder = SimpleRNN(latent_dim, return_state=True)
encoder_outputs, state_h = encoder(encoder_inputs)
encoder_states = [state_h]

# Decoder
decoder_inputs = Input(shape=(None, len(target_characters)))
decoder_rnn = SimpleRNN(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_rnn(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(len(target_characters), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
tqdm_callback = tfa.callbacks.TQDMProgressBar()
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
          callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models"),tqdm_callback])

model.save("models/"+model_name)
wandb.finish()