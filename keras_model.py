import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, TimeDistributed, Embedding, Dropout, Bidirectional, Attention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from get_data import load_and_prepare_data, get_data_by_amount
import nltk
nltk.download('words')
from nltk.corpus import words
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import math
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np

def is_english(text):
    english_words = set(words.words())
    text_words = set(text.lower().split())
    count_true = len(text_words.intersection(english_words))
    return count_true >= 3 or count_true == len(text_words)


def create_improved_model(max_frames, rgb_features, audio_features, vocab_size, max_title_length, embedding_dim=300):
    rgb_input = Input(shape=(max_frames, rgb_features))
    audio_input = Input(shape=(max_frames, audio_features))
    decoder_input = Input(shape=(max_title_length,))

    # Improved feature merger
    merged_features = Concatenate(axis=-1)([rgb_input, audio_input])
    merged_features = TimeDistributed(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))(merged_features)
    merged_features = LayerNormalization()(merged_features)
    merged_features = Dropout(0.2)(merged_features)

    # Encoder with residual connections
    encoder = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.01)))(merged_features)
    encoder = LayerNormalization()(encoder)
    encoder = Dropout(0.2)(encoder)
    encoder_residual = encoder + merged_features
    encoder = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.01)))(encoder_residual)
    encoder = LayerNormalization()(encoder)
    encoder = Dropout(0.2)(encoder)

    # Attention mechanism
    attention = Attention()([encoder, encoder])

    # Flatten the attention output
    attention_flat = tf.keras.layers.Flatten()(attention)

    # Dense layer to match dimensions
    attention_dense = Dense(512, activation='relu')(attention_flat)

    # Decoder
    embedding = Embedding(vocab_size, embedding_dim, embeddings_regularizer=l2(0.01))(decoder_input)
    decoder = LSTM(512, return_sequences=True, kernel_regularizer=l2(0.01))
    decoder_outputs = decoder(embedding, initial_state=[attention_dense, attention_dense])
    decoder_outputs = LayerNormalization()(decoder_outputs)
    decoder_outputs = Dropout(0.2)(decoder_outputs)

    output = TimeDistributed(Dense(vocab_size, activation='softmax', kernel_regularizer=l2(0.01)))(decoder_outputs)

    model = Model(inputs=[rgb_input, audio_input, decoder_input], outputs=output)
    return model
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def calculate_bleu(y_true, y_pred, tokenizer):
    smooth = SmoothingFunction().method1
    bleu_scores = []
    for true, pred in zip(y_true, y_pred):
        true_sentence = [tokenizer.index_word.get(idx, '') for idx in true if idx != 0]
        pred_sentence = [tokenizer.index_word.get(idx, '') for idx in pred if idx != 0]
        bleu_scores.append(sentence_bleu([true_sentence], pred_sentence, smoothing_function=smooth))
    return np.mean(bleu_scores)

def calculate_perplexity(y_true, y_pred):
    epsilon = 1e-10
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = tf.exp(tf.reduce_mean(cross_entropy))
    return tf.minimum(perplexity, 1e6)  # Cap perplexity to avoid numerical instability

class BLEUCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, tokenizer):
        super(BLEUCallback, self).__init__()
        self.validation_data = validation_data
        self.tokenizer = tokenizer
        self.bleu_scores = []

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val)
        y_pred_indices = np.argmax(y_pred, axis=-1)
        bleu = calculate_bleu(y_val, y_pred_indices, self.tokenizer)
        self.bleu_scores.append(bleu)
        logs['bleu_score'] = bleu
        print(f' - BLEU score: {bleu:.4f}')

def train_improved_model(X_rgb, X_audio, y, tokenizer, max_title_length, epochs=100, batch_size=128):
    max_frames, rgb_features = X_rgb.shape[1], X_rgb.shape[2]
    audio_features = X_audio.shape[2]
    vocab_size = len(tokenizer.word_index) + 1

    model = create_improved_model(max_frames, rgb_features, audio_features, vocab_size, max_title_length)

    # Use a fixed initial learning rate
    initial_learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    model.compile(optimizer=optimizer,
                  loss=calculate_perplexity,
                  metrics=['accuracy'])

    decoder_input = np.zeros_like(y)
    decoder_input[:, 1:] = y[:, :-1]
    decoder_input[:, 0] = tokenizer.word_index['<start>']

    val_split = 0.2
    split_index = int(len(X_rgb) * (1 - val_split))
    X_rgb_train, X_rgb_val = X_rgb[:split_index], X_rgb[split_index:]
    X_audio_train, X_audio_val = X_audio[:split_index], X_audio[split_index:]
    decoder_input_train, decoder_input_val = decoder_input[:split_index], decoder_input[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    bleu_callback = BLEUCallback(([X_rgb_val, X_audio_val, decoder_input_val], y_val), tokenizer)

    def print_prediction(epoch, logs):
        if epoch % 1 == 0:  # Print every epoch
            test_indices = np.random.choice(len(X_rgb_val), 3, replace=False)
            test_rgb = X_rgb_val[test_indices]
            test_audio = X_audio_val[test_indices]
            test_y = y_val[test_indices]

            for i in range(3):
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

    callbacks = [
        ModelCheckpoint('best_model.keras', save_best_only=True, monitor='bleu_score', mode='max'),
        bleu_callback,
        LambdaCallback(on_epoch_end=print_prediction),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
        EarlyStopping(patience=15, restore_best_weights=True)
    ]

    history = model.fit(
        [X_rgb_train, X_audio_train, decoder_input_train],
        y_train,
        validation_data=([X_rgb_val, X_audio_val, decoder_input_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    return model, history, bleu_callback.bleu_scores
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

    bleu = calculate_bleu(y, input_seq, tokenizer)
    print(f"BLEU score: {bleu:.4f}")

    plt.figure(figsize=(10, 5))
    plt.imshow(predictions[0].T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.title('Prediction Probabilities for Each Word')
    plt.xlabel('Time Step')
    plt.ylabel('Vocabulary Index')
    plt.tight_layout()
    plt.show()

def main():
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

    tokenizer.word_index['<start>'] = max(list(tokenizer.word_index.values())) + 1
    tokenizer.word_index['<end>'] = max(list(tokenizer.word_index.values())) + 1
    tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}

    y_with_tokens = np.zeros((y.shape[0], y.shape[1] + 2))
    y_with_tokens[:, 1:-1] = y
    y_with_tokens[:, 0] = tokenizer.word_index['<start>']
    y_with_tokens[:, -1] = tokenizer.word_index['<end>']
    y = y_with_tokens

    max_title_length = y.shape[1]

    model, history, bleu_scores = train_improved_model(X_rgb, X_audio, y, tokenizer, max_title_length)

    model.summary()

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(132)
    plt.plot(history.history['loss'], label='Train Perplexity')
    plt.plot(history.history['val_loss'], label='Validation Perplexity')
    plt.title('Model Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()

    plt.subplot(133)
    plt.plot(bleu_scores, label='BLEU Score')
    plt.title('BLEU Score')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU')
    plt.legend()

    plt.tight_layout()
    plt.show()

    test_rgb = X_rgb[:1]
    test_audio = X_audio[:1]
    predict_and_analyze(model, test_rgb, test_audio, y[:1], tokenizer, max_title_length)

if __name__ == "__main__":
    main()
