from get_data import load_and_prepare_data
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Concatenate, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import RepeatVector
import matplotlib.pyplot as plt



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
    y_reshaped = y[:, :, np.newaxis]

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', save_best_only=True)
    ]

    class PredictionCallback(tf.keras.callbacks.Callback):
        def __init__(self, X_rgb, X_audio, y, tokenizer, num_samples=5):
            self.X_rgb = X_rgb[:num_samples]
            self.X_audio = X_audio[:num_samples]
            self.y = y[:num_samples]  # Use the original y, not the reshaped one
            self.tokenizer = tokenizer
            self.num_samples = num_samples

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 3 == 0:  # Print every 3 epochs
                predictions = self.model.predict([self.X_rgb, self.X_audio])
                print(f"\nPredictions vs Real Values at Epoch {epoch + 1}:")
                for i in range(self.num_samples):
                    predicted_sequence = np.argmax(predictions[i], axis=1)
                    predicted_title = ' '.join([self.tokenizer.index_word.get(idx, '') for idx in predicted_sequence if idx != 0])
                    real_title = ' '.join([self.tokenizer.index_word.get(idx, '') for idx in self.y[i] if idx != 0])
                    print(f"Sample {i + 1}:")
                    print(f"Predicted: {predicted_title}")
                    print(f"Real: {real_title}")
                    print()

    prediction_callback = PredictionCallback(X_rgb, X_audio, y, tokenizer)  # Use original y
    callbacks.append(prediction_callback)

    # Train model
    history = model.fit(
        [X_rgb, X_audio],
        y_reshaped,  # Use the reshaped y for training
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
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



if __name__ == "__main__":
    main()