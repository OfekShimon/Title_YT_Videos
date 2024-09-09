from get_data import load_and_prepare_data
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Concatenate, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import RepeatVector


def create_model(max_frames, rgb_features, audio_features, vocab_size, max_title_length):
    # Input layers
    rgb_input = Input(shape=(max_frames, rgb_features))
    audio_input = Input(shape=(max_frames, audio_features))

    # Combine RGB and audio features
    combined_input = Concatenate()([rgb_input, audio_input])

    # Encoder
    encoder = Bidirectional(LSTM(256, return_sequences=True))(combined_input)
    encoder = Bidirectional(LSTM(128))(encoder)

    # Decoder
    decoder = RepeatVector(max_title_length)(encoder)
    decoder = LSTM(256, return_sequences=True)(decoder)

    # Output layer
    output = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder)

    # Create model
    model = Model(inputs=[rgb_input, audio_input], outputs=output)

    return model


def train_model(X_rgb, X_audio, y, tokenizer, max_title_length, epochs=50, batch_size=32):
    max_frames, rgb_features = X_rgb.shape[1], X_rgb.shape[2]
    audio_features = X_audio.shape[2]
    vocab_size = len(tokenizer.word_index) + 1

    # Create model
    model = create_model(max_frames, rgb_features, audio_features, vocab_size, max_title_length)

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Reshape y to add a dimension for sparse_categorical_crossentropy
    y = y[:, :, np.newaxis]

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

    # Train model
    history = model.fit(
        [X_rgb, X_audio],
        y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint]
    )

    return model, history


# Usage
def main():
    # Load and prepare data
    X_rgb, X_audio, y, tokenizer = load_and_prepare_data()
    max_title_length = y.shape[1]

    # Train model
    model, history = train_model(X_rgb, X_audio, y, tokenizer, max_title_length)

    # Print model summary
    model.summary()

    # Plot training history
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
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

    # Test the model
    test_rgb = X_rgb[:1]  # Take the first sample for testing
    test_audio = X_audio[:1]
    predicted_sequence = model.predict([test_rgb, test_audio])
    predicted_words = np.argmax(predicted_sequence[0], axis=1)
    predicted_title = ' '.join([tokenizer.index_word[idx] for idx in predicted_words if idx != 0])
    print("Predicted title:", predicted_title)
    print("Actual title:", ' '.join([tokenizer.index_word[idx] for idx in y[0] if idx != 0]))


# if __name__ == "__main__":
#     main()
#
#     # Train model
#     model, history = train_model(X_rgb, X_audio, y, tokenizer)
#
#     # Print model summary
#     model.summary()

    # # Plot training history
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(10, 4))
    # plt.subplot(121)
    # plt.plot(history.history['accuracy'], label='Train Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.title('Model Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    #
    # plt.subplot(122)
    # plt.plot(history.history['loss'], label='Train Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()