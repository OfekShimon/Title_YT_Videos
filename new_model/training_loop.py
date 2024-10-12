import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def greedy_decode(model, input_seq, tokenizer, max_length):
    # Initialize target sequence with start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['startseq']

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens = model.predict([np.expand_dims(input_seq, 0), target_seq], verbose=0)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tokenizer.index_word.get(sampled_token_index, '<OOV>')
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length or find stop character.
        if sampled_char == 'endseq' or len(decoded_sentence) > max_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.concatenate([target_seq, [[sampled_token_index]]], axis=-1)

    return decoded_sentence


def evaluate_model(model, val_dataset, tokenizer, max_length):
    predictions = []
    references = []
    print("Evaluating model...")
    for (encoder_input, decoder_input), decoder_target in val_dataset:
        # Predict in batch
        batch_size = encoder_input.shape[0]
        decoder_input = np.zeros((batch_size, 1))
        decoder_input[:, 0] = tokenizer.word_index['startseq']

        for _ in range(max_length):

            output = model.predict([encoder_input, decoder_input], verbose=0)
            sampled_token_indices = np.argmax(output[:, -1, :], axis=-1)
            decoder_input = np.concatenate([decoder_input, sampled_token_indices.reshape(-1, 1)], axis=-1)

        # Process predictions and references
        for i in range(batch_size):
            pred_seq = [tokenizer.index_word.get(idx, '<OOV>') for idx in decoder_input[i, 1:]]
            pred_text = [token for token in pred_seq if token not in ['startseq', 'endseq', '<OOV>']]
            true_text = [tokenizer.index_word.get(j, '') for j in decoder_target[i].numpy() if
                         j != 0 and j != tokenizer.word_index['endseq']]

            # Stop at 'endseq' token for predictions
            if 'endseq' in pred_text:
                pred_text = pred_text[:pred_text.index('endseq')]

            predictions.append(pred_text)
            references.append([true_text])  # Wrap true_text in a list for BLEU calculation

    # Calculate BLEU score
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    for ref, hyp in zip(references, predictions):
        if len(hyp) == 0:
            continue  # Skip empty predictions
        bleu_scores.append(sentence_bleu(ref, hyp, smoothing_function=smoothie))

    bleu_score = np.mean(bleu_scores) if bleu_scores else 0.0

    print(f"BLEU score on validation set: {bleu_score:.4f}")

    # Print some sample predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(predictions))):
        print(f"Sample {i + 1}:")
        print(f"Actual: {' '.join(references[i][0])}")
        print(f"Predicted: {' '.join(predictions[i])}\n")

    return bleu_score

class BleuCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, tokenizer, max_length):
        super(BleuCallback, self).__init__()
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bleu_scores = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 20 == 0:
            bleu_score = evaluate_model(self.model, self.val_data.take(10), self.tokenizer, self.max_length)
            print(f'\nEpoch {epoch + 1} - BLEU score: {bleu_score:.4f}')
            self.bleu_scores.append(bleu_score)


def train_model(model, train_dataset, val_dataset, tokenizer, epochs=50, max_length=20):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    bleu_callback = BleuCallback(val_dataset, tokenizer, max_length)

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[bleu_callback],
        verbose=1
    )

    return model, history, bleu_callback.bleu_scores