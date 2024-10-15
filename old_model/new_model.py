import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
import matplotlib.pyplot as plt
from get_data import load_and_prepare_data, get_data_by_amount
import nltk
nltk.download('words')
from nltk.corpus import words
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import math

def is_english(text):
    english_words = set(words.words())
    text_words = set(text.lower().split())
    count_true = len(text_words.intersection(english_words))
    return count_true >= 3 or count_true == len(text_words)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    if mask is not None:
        logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, factor=0.2, patience=5, min_lr=1e-6):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_val_loss = float('inf')
        self.wait = 0
        self.reduced_lr = 1.0

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(tf.cast(self.d_model, tf.float32)) * tf.math.minimum(arg1, arg2)
        return tf.maximum(lr * self.reduced_lr, self.min_lr)

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.reduced_lr *= self.factor
                self.wait = 0
                print(f"\nReducing learning rate to {self.reduced_lr:.6f}")

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "factor": self.factor,
            "patience": self.patience,
            "min_lr": self.min_lr
        }

def calculate_bleu(y_true, y_pred, tokenizer):
    smooth = SmoothingFunction().method1
    bleu_scores = []
    for true, pred in zip(y_true, y_pred):
        true_sentence = [tokenizer.index_word.get(idx, '') for idx in true if idx != 0]
        pred_sentence = [tokenizer.index_word.get(idx, '') for idx in pred if idx != 0]
        bleu_scores.append(sentence_bleu([true_sentence], pred_sentence, smoothing_function=smooth))
    return np.mean(bleu_scores)

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

class LearningRateSchedulerCallback(tf.keras.callbacks.Callback):
    def __init__(self, schedule):
        super(LearningRateSchedulerCallback, self).__init__()
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs=None):
        self.schedule.on_epoch_end(epoch, logs)

def create_improved_model(max_frames, rgb_features, audio_features, vocab_size, max_title_length, d_model=512,
                          num_layers=6, num_heads=8, dff=2048, dropout_rate=0.1):
    inputs = Input(shape=(max_frames, rgb_features + audio_features))
    target = Input(shape=(max_title_length - 1,))  # Adjusted for teacher forcing

    # Encoder
    enc = Dense(d_model)(inputs)
    enc = PositionalEncoding(max_frames, d_model)(enc)
    enc = Dropout(dropout_rate)(enc)

    for _ in range(num_layers):
        enc = MultiHeadAttentionLayer(d_model, num_heads)(enc, enc, enc, None)
        enc = LayerNormalization(epsilon=1e-6)(enc)
        ff = Dense(dff, activation='relu')(enc)
        ff = Dense(d_model)(ff)
        enc = enc + ff
        enc = Dropout(dropout_rate)(enc)
        enc = LayerNormalization(epsilon=1e-6)(enc)

    # Decoder
    dec = Embedding(vocab_size, d_model)(target)
    dec = PositionalEncoding(max_title_length - 1, d_model)(dec)  # Adjusted for teacher forcing
    dec = Dropout(dropout_rate)(dec)

    for _ in range(num_layers):
        look_ahead_mask = tf.keras.layers.Lambda(
            lambda x: create_look_ahead_mask(tf.shape(x)[1]))(dec)
        dec = MultiHeadAttentionLayer(d_model, num_heads)(dec, dec, dec, look_ahead_mask)
        dec = LayerNormalization(epsilon=1e-6)(dec)

        dec = MultiHeadAttentionLayer(d_model, num_heads)(enc, enc, dec, None)
        dec = LayerNormalization(epsilon=1e-6)(dec)

        ff = Dense(dff, activation='relu')(dec)
        ff = Dense(d_model)(ff)
        dec = dec + ff
        dec = Dropout(dropout_rate)(dec)
        dec = LayerNormalization(epsilon=1e-6)(dec)

    outputs = Dense(vocab_size, activation='softmax')(dec)
    model = Model([inputs, target], outputs)
    return model


def train_improved_model(X_rgb, X_audio, y, tokenizer, max_title_length, epochs=100, batch_size=64):
    max_frames, rgb_features = X_rgb.shape[1], X_rgb.shape[2]
    audio_features = X_audio.shape[2]
    vocab_size = len(tokenizer.word_index) + 1

    X_combined = np.concatenate([X_rgb, X_audio], axis=-1)

    model = create_improved_model(max_frames, rgb_features, audio_features, vocab_size, max_title_length)

    learning_rate = CustomSchedule(512, factor=0.2, patience=5, min_lr=1e-6)
    optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_object = SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    val_split = 0.2
    split_index = int(len(X_combined) * (1 - val_split))
    X_train, X_val = X_combined[:split_index], X_combined[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    # Adjust target sequences for training
    y_train_input = y_train[:, :-1]
    y_train_target = y_train[:, 1:]
    y_val_input = y_val[:, :-1]
    y_val_target = y_val[:, 1:]

    bleu_callback = BLEUCallback(([X_val, y_val_input], y_val_target), tokenizer)

    def print_prediction(epoch, logs):
        if epoch % 1 == 0:  # Print every epoch
            test_indices = np.random.choice(len(X_val), 3, replace=False)
            test_X = X_val[test_indices]
            test_y = y_val[test_indices]

            for i in range(3):
                input_seq = np.zeros((1, max_title_length - 1))
                input_seq[0, 0] = tokenizer.word_index['<start>']

                for j in range(1, max_title_length - 1):
                    predictions = model.predict([test_X[i:i + 1], input_seq])
                    predicted_id = np.argmax(predictions[0, j - 1])
                    input_seq[0, j] = predicted_id
                    if predicted_id == tokenizer.word_index.get('<end>', 1):
                        break

                predicted_words = [tokenizer.index_word.get(idx, '') for idx in input_seq[0] if idx != 0]
                predicted_title = ' '.join(predicted_words[1:])  # Remove start token
                actual_title = ' '.join([tokenizer.index_word.get(idx, '') for idx in test_y[i] if idx != 0])
                print(f"\nEpoch {epoch} - Sample {i + 1}")
                print(f"Predicted: {predicted_title}")
                print(f"Actual: {actual_title}")

    callbacks = [
        ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min'),
        bleu_callback,
        LambdaCallback(on_epoch_end=print_prediction),
        LearningRateSchedulerCallback(learning_rate),
        EarlyStopping(patience=15, restore_best_weights=True)
    ]

    history = model.fit(
        [X_train, y_train_input],
        y_train_target,
        validation_data=([X_val, y_val_input], y_val_target),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    return model, history, bleu_callback.bleu_scores

def predict_and_analyze(model, X_rgb, X_audio, y, tokenizer, max_title_length):
    X_combined = np.concatenate([X_rgb, X_audio], axis=-1)
    input_seq = np.zeros((1, max_title_length))
    input_seq[0, 0] = tokenizer.word_index['<start>']

    for i in range(1, max_title_length):
        predictions = model.predict([X_combined, input_seq])
        predicted_id = np.argmax(predictions[0, i-1])
        input_seq[0, i] = predicted_id
        if predicted_id == tokenizer.word_index.get('<end>', 1):
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
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
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