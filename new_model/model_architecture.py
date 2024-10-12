import tensorflow as tf

def create_seq2seq_model(vocab_size, max_input_length, max_output_length, embed_dim=256, lstm_units=256, dropout_rate=0.1):
    # Encoder
    encoder_inputs = tf.keras.layers.Input(shape=(max_input_length,))
    encoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = tf.keras.layers.LSTM(lstm_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = tf.keras.layers.Input(shape=(max_output_length-1,))  # Changed this line
    decoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
    decoder_lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

    decoder_embedded = decoder_embedding(decoder_inputs)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
    decoder_outputs = tf.keras.layers.Dropout(dropout_rate)(decoder_outputs)
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Inference mode (sampling)
    encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = tf.keras.layers.Input(shape=(lstm_units,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(lstm_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedded_inf = decoder_embedding(decoder_inputs)
    decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(decoder_embedded_inf, initial_state=decoder_states_inputs)
    decoder_states_inf = [state_h_inf, state_c_inf]
    decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

    decoder_model = tf.keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs_inf] + decoder_states_inf)

    return model, encoder_model, decoder_model


if __name__ == "__main__":
    model, encoder_model, decoder_model = create_seq2seq_model(vocab_size=20000, max_input_length=500, max_output_length=20)
    model.summary()
    encoder_model.summary()
    decoder_model.summary()