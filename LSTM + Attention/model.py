from keras.layers import (
    Input,
    LSTM,
    Embedding,
    Dense,
    Bidirectional,
    Concatenate,
    Dot,
    Activation,
    TimeDistributed,
)
from keras.models import Model
from keras.utils import plot_model
from dataPreprocessorWordLevel import num_encoder_tokens, num_decoder_tokens


def seq2seq_Attention(num_encoder_tokens, num_decoder_tokens, embd_sz, latent_dim):
    encoder_inputs = Input(shape=(None,), dtype="float32")
    encoder_inputs_ = Embedding(num_encoder_tokens, embd_sz, mask_zero=True)(
        encoder_inputs
    )

    encoder = Bidirectional(LSTM(latent_dim, return_state=True, return_sequences=True))
    encoder_outputs, state_f_h, state_f_c, sate_b_h, state_b_c = encoder(encoder_inputs_)

    # We discard `encoder_outputs` and only keep the states.
    state_h = Concatenate()([state_f_h, sate_b_h])
    state_c = Concatenate()([state_f_c, state_b_c])

    encoder_states = [state_h, state_c]
    # print(encoder_states)

    decoder_inputs = Input(shape=(None,))
    decoder_inputs_ = Embedding(num_decoder_tokens, embd_sz, mask_zero=True)(
        decoder_inputs
    )

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.

    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs_, initial_state=encoder_states)

    # print(decoder_outputs)
    # print(encoder_outputs)

    att_dot = Dot(axes=[2, 2])
    attention = att_dot([decoder_outputs, encoder_outputs])
    att_activation = Activation(activation="softmax", name="attention")

    attention = att_activation(attention)
    # print("Attention", attention)

    context_dot = Dot(axes=[2, 1])
    context = context_dot([attention, encoder_outputs])
    att_context_concat = Concatenate()
    decoder_combined_context = att_context_concat([context, decoder_outputs])

    decoder_dense = Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_combined_context)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # print(model.summary())

    encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

    decoder_encoder_inputs = Input(
        shape=(
            None,
            latent_dim * 2,
        )
    )
    decoder_state_input_h = Input(shape=(latent_dim * 2,))  # Bi LSTM
    decoder_state_input_c = Input(shape=(latent_dim * 2,))  # Bi LSTM

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs_, initial_state=decoder_states_inputs
    )

    decoder_states = [state_h, state_c]

    attention = att_dot([decoder_outputs, decoder_encoder_inputs])
    attention = att_activation(attention)
    context = context_dot([attention, decoder_encoder_inputs])

    decoder_combined_context = att_context_concat([context, decoder_outputs])

    decoder_outputs = decoder_dense(decoder_combined_context)

    decoder_model = Model(
        [decoder_inputs, decoder_encoder_inputs] + decoder_states_inputs,
        [decoder_outputs, attention] + decoder_states,
    )

    return model, encoder_model, decoder_model

embd_sz= 100

model,encoder_model,decoder_model =seq2seq_Attention(num_encoder_tokens,num_decoder_tokens,embd_sz=embd_sz,latent_dim=embd_sz)
# print(model.summary())

# plot_model(model,show_shapes=True,show_layer_names=True)

