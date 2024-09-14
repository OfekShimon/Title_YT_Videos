import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Concatenate, Bidirectional, RepeatVector, \
    Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
import matplotlib.pyplot as plt
from get_data import load_and_prepare_data, get_data_by_amount


def create_model(max_frames, rgb_features, audio_features, vocab_size, max_title_length):
    # Input layers
    rgb_input = Input(shape=(max_frames, rgb_features))
    audio_input = Input(shape=(max_frames, audio_features))

    # Combine RGB and audio features
    combined_input = Concatenate()([rgb_input, audio_input])

    # Encoder
    encoder = Bidirectional(LSTM(256, return_sequences=True))(combined_input)
    encoder = LayerNormalization()(encoder)
    encoder = Dropout(0.3)(encoder)
    encoder = Bidirectional(LSTM(128))(encoder)
    encoder = LayerNormalization()(encoder)
    encoder = Dropout(0.3)(encoder)

    # Decoder
    decoder = RepeatVector(max_title_length)(encoder)
    decoder = LSTM(256, return_sequences=True)(decoder)
    decoder = LayerNormalization()(decoder)
    decoder = Dropout(0.3)(decoder)

    # Output layer
    output = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder)

    # Create model
    model = Model(inputs=[rgb_input, audio_input], outputs=output)

    return model


def custom_loss(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def custom_accuracy(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    y_pred_argmax = tf.argmax(y_pred, axis=-1)
    correct = tf.equal(tf.cast(y_true, dtype=tf.int64), y_pred_argmax)
    mask = tf.cast(mask, dtype=tf.float32)
    correct = tf.cast(correct, dtype=tf.float32) * mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)


def train_model(X_rgb, X_audio, y, tokenizer, max_title_length, epochs=100, batch_size=32):
    max_frames, rgb_features = X_rgb.shape[1], X_rgb.shape[2]
    audio_features = X_audio.shape[2]
    vocab_size = len(tokenizer.word_index) + 1

    # Create model
    model = create_model(max_frames, rgb_features, audio_features, vocab_size, max_title_length)

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=[custom_accuracy])

    # Print prediction callback
    def print_prediction(epoch, logs):
        if epoch % 10 == 0:
            test_rgb = X_rgb[:1]
            test_audio = X_audio[:1]
            predicted_sequence = model.predict([test_rgb, test_audio])
            predicted_words = np.argmax(predicted_sequence[0], axis=1)
            predicted_title = ' '.join([tokenizer.index_word.get(idx, '') for idx in predicted_words if idx != 0])
            print(f"\nEpoch {epoch} - Predicted title: {predicted_title}")

    # Define callbacks
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', save_best_only=True),
        LambdaCallback(on_epoch_end=print_prediction)
    ]

    # Train model
    history = model.fit(
        [X_rgb, X_audio],
        y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    return model, history


def predict_and_analyze(model, X_rgb, X_audio, y, tokenizer):
    predicted_sequence = model.predict([X_rgb, X_audio])

    print("Raw prediction shape:", predicted_sequence.shape)
    print("Raw prediction sample (first 5 time steps, first 10 vocab):")
    print(predicted_sequence[0, :5, :10])

    predicted_words = np.argmax(predicted_sequence[0], axis=1)
    print("Predicted word indices:", predicted_words)

    predicted_title = ' '.join([tokenizer.index_word.get(idx, '') for idx in predicted_words if idx != 0])
    actual_title = ' '.join([tokenizer.index_word.get(idx, '') for idx in y[0] if idx != 0])

    print("Predicted title:", predicted_title)
    print("Actual title:", actual_title)

    # Analyze prediction distribution
    plt.figure(figsize=(10, 5))
    plt.imshow(predicted_sequence[0].T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.title('Prediction Probabilities for Each Word')
    plt.xlabel('Time Step')
    plt.ylabel('Vocabulary Index')
    plt.tight_layout()
    plt.show()


def main():
    # Load and prepare train data
    data_amount = 100
    train_data_path = f'train_input_output_data_{data_amount}.pkl'
    if not os.path.exists(train_data_path):
        print("preprocessing data...")
        get_data_by_amount(data_amount, 'train')
    X_rgb, X_audio, y, tokenizer = load_and_prepare_data(train_data_path)

    print("X_rgb shape:", X_rgb.shape)
    print("X_audio shape:", X_audio.shape)
    print("y shape:", y.shape)
    print("Vocabulary size:", len(tokenizer.word_index) + 1)

    max_title_length = y.shape[1]

    # Train model
    model, history = train_model(X_rgb, X_audio, y, tokenizer, max_title_length)

    # Print model summary
    model.summary()

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['custom_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_custom_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Make predictions on a test sample
    test_rgb = X_rgb[:1]
    test_audio = X_audio[:1]
    predict_and_analyze(model, test_rgb, test_audio, y[:1], tokenizer)


if __name__ == "__main__":
    main()