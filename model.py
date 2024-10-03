import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, TimeDistributed, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
import matplotlib.pyplot as plt
from get_data import load_and_prepare_data, get_data_by_amount
import enchant


def create_model(max_frames, rgb_features, audio_features, vocab_size, max_title_length, embedding_dim=256):
    # Input layers
    rgb_input = Input(shape=(max_frames, rgb_features))
    audio_input = Input(shape=(max_frames, audio_features))
    decoder_input = Input(shape=(max_title_length,))

    # Merge RGB and audio features
    merged_features = Concatenate(axis=-1)([rgb_input, audio_input])

    # Optional: Add a dense layer to learn joint representations
    merged_features = TimeDistributed(Dense(512, activation='relu'))(merged_features)

    # Encoder
    encoder = LSTM(512, return_sequences=True)(merged_features)
    encoder = LSTM(512)(encoder)

    # Decoder
    embedding = Embedding(vocab_size, embedding_dim)(decoder_input)
    decoder = LSTM(512, return_sequences=True)(embedding, initial_state=[encoder, encoder])
    decoder = Dropout(0.5)(decoder)
    output = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder)

    model = Model(inputs=[rgb_input, audio_input, decoder_input], outputs=output)
    return model


def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def train_model(X_rgb, X_audio, y, tokenizer, max_title_length, epochs=100, batch_size=64):
    max_frames, rgb_features = X_rgb.shape[1], X_rgb.shape[2]
    audio_features = X_audio.shape[2]
    vocab_size = len(tokenizer.word_index) + 1

    # Create model
    model = create_model(max_frames, rgb_features, audio_features, vocab_size, max_title_length)

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Prepare decoder input (with start token)
    decoder_input = np.zeros_like(y)
    decoder_input[:, 1:] = y[:, :-1]
    decoder_input[:, 0] = tokenizer.word_index['<start>']

    # Print prediction callback
    def print_prediction(epoch, logs):
        if epoch % 1 == 0:  # Print every epoch
            test_indices = np.random.choice(len(X_rgb), 10, replace=False)
            test_rgb = X_rgb[test_indices]
            test_audio = X_audio[test_indices]
            test_y = y[test_indices]

            for i in range(10):
                input_seq = np.zeros((1, max_title_length))
                input_seq[0, 0] = tokenizer.word_index['<start>']

                for j in range(1, max_title_length):
                    predictions = model.predict([test_rgb[i:i + 1], test_audio[i:i + 1], input_seq])
                    sampled_token = sample_with_temperature(predictions[0, j - 1], temperature=0.7)
                    input_seq[0, j] = sampled_token
                    if sampled_token == tokenizer.word_index.get('<end>', 1):
                        break

                predicted_words = [tokenizer.index_word.get(idx, '') for idx in input_seq[0] if idx != 0]
                predicted_title = ' '.join(predicted_words[1:-1])  # Remove start and end tokens
                actual_title = ' '.join([tokenizer.index_word.get(idx, '') for idx in test_y[i] if idx != 0])
                print(f"\nEpoch {epoch} - Sample {i + 1}")
                print(f"Predicted: {predicted_title}")
                print(f"Actual: {actual_title}")

    # Define callbacks
    callbacks = [
        # EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', save_best_only=True),
        LambdaCallback(on_epoch_end=print_prediction)
    ]

    # Train model
    history = model.fit(
        [X_rgb, X_audio, decoder_input],
        y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    return model, history


def predict_and_analyze(model, X_rgb, X_audio, y, tokenizer, max_title_length):
    input_seq = np.zeros((1, max_title_length))
    input_seq[0, 0] = tokenizer.word_index['<start>']

    for i in range(1, max_title_length):
        predictions = model.predict([X_rgb, X_audio, input_seq])
        sampled_token = sample_with_temperature(predictions[0, i - 1], temperature=0.7)
        input_seq[0, i] = sampled_token
        if sampled_token == tokenizer.word_index.get('<end>', 1):
            break

    predicted_words = [tokenizer.index_word.get(idx, '') for idx in input_seq[0] if idx != 0]
    predicted_title = ' '.join(predicted_words[1:-1])  # Remove start and end tokens

    print("Raw prediction shape:", predictions.shape)
    print("Raw prediction sample (first 5 time steps, first 10 vocab):")
    print(predictions[0, :5, :10])

    print("Predicted word indices:", input_seq[0])

    actual_title = ' '.join([tokenizer.index_word.get(idx, '') for idx in y[0] if idx != 0])

    print("Predicted title:", predicted_title)
    print("Actual title:", actual_title)

    # Analyze prediction distribution
    plt.figure(figsize=(10, 5))
    plt.imshow(predictions[0].T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.title('Prediction Probabilities for Each Word')
    plt.xlabel('Time Step')
    plt.ylabel('Vocabulary Index')
    plt.tight_layout()
    plt.show()


def is_english(text):
    text = text.split()
    dictionary = enchant.Dict("en_US")
    count_true = 0
    count_false = 0
    for i in range(len(text)):
        if dictionary.check(text[i]):
            count_true += 1
        else:
            count_false += 1
    return count_true >= 4 or count_false == 0


def main():
    # Load and prepare train data
    data_amount = 10000
    train_data_path = "merged_train_data.csv"
    if not os.path.exists(train_data_path):
        print("preprocessing data...")
        data_filter = lambda data_video: is_english(data_video.get('title', '')) and data_video.get('view_count', 0) > 100
        get_data_by_amount(data_amount, 'train', train_data_path, data_filter)
    X_rgb, X_audio, y, tokenizer = load_and_prepare_data(train_data_path)

    print("X_rgb shape:", X_rgb.shape)
    print("X_audio shape:", X_audio.shape)
    print("y shape:", y.shape)
    print("Vocabulary size:", len(tokenizer.word_index) + 1)

    # Ensure '<start>' and '<end>' tokens are in the vocabulary
    tokenizer.word_index['<start>'] = max(list(tokenizer.word_index.values())) + 1
    tokenizer.word_index['<end>'] = max(list(tokenizer.word_index.values())) + 1
    tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}

    # Add start and end tokens to y
    y_with_tokens = np.zeros((y.shape[0], y.shape[1] + 2))
    y_with_tokens[:, 1:-1] = y
    y_with_tokens[:, 0] = tokenizer.word_index['<start>']
    y_with_tokens[:, -1] = tokenizer.word_index['<end>']
    y = y_with_tokens

    max_title_length = y.shape[1]

    # Train model
    model, history = train_model(X_rgb, X_audio, y, tokenizer, max_title_length)

    # Print model summary
    model.summary()

    # Plot training history
    plt.figure(figsize=(12, 4))
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

    # Make predictions on a test sample
    test_rgb = X_rgb[:1]
    test_audio = X_audio[:1]
    predict_and_analyze(model, test_rgb, test_audio, y[:1], tokenizer, max_title_length)


if __name__ == "__main__":
    main()