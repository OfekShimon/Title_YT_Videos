import os
import argparse
import numpy as np
import tensorflow as tf
from data_preprocessing import preprocess_video_data
from model_architecture import create_seq2seq_model
from training_loop import train_model, evaluate_model
import pickle
from plot_metrics import plot_metrics


def prepare_data(encoder_input, decoder_target):
    # Ensure encoder_input is 2D
    encoder_input = tf.squeeze(encoder_input)
    if len(tf.shape(encoder_input)) == 1:
        encoder_input = tf.expand_dims(encoder_input, axis=0)

    # Ensure decoder_target is 2D
    decoder_target = tf.squeeze(decoder_target)
    if len(tf.shape(decoder_target)) == 1:
        decoder_target = tf.expand_dims(decoder_target, axis=0)

    decoder_input = decoder_target[:, :-1]  # everything except the last token
    decoder_output = decoder_target[:, 1:]  # everything except the first token
    return (encoder_input, decoder_input), decoder_output

def main(args):
    # Step 1: Preprocess data (unchanged)
    if not os.path.exists(args.preprocessed_data_file) or True:
        print("Preprocessing data...")
        preprocess_video_data(args.input_file, args.preprocessed_data_file, args.max_transcript_length,
                              args.max_title_length+1, args.num_samples, args.vocab_size)
    else:
        print("Preprocessed data file already exists. Skipping preprocessing.")

    # Step 2: Load preprocessed data
    with np.load(args.preprocessed_data_file, allow_pickle=True) as data:
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']

    # Load tokenizer
    with open(args.preprocessed_data_file + "_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(prepare_data)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.map(prepare_data)

    # Debug: print shapes before batching
    print("Sample shapes before batching:")
    for (encoder_input, decoder_input), decoder_output in train_dataset.take(1):
        print(f"Encoder input shape: {encoder_input.shape}")
        print(f"Decoder input shape: {decoder_input.shape}")
        print(f"Decoder output shape: {decoder_output.shape}")

    # After creating the datasets, update the batching process:
    train_dataset = train_dataset.batch(args.batch_size)
    val_dataset = val_dataset.batch(args.batch_size)

    # Add these debug prints after batching
    print("Sample shapes after batching:")
    for (encoder_input, decoder_input), decoder_output in train_dataset.take(1):
        print(f"Encoder input shape: {encoder_input.shape}")
        print(f"Decoder input shape: {decoder_input.shape}")
        print(f"Decoder output shape: {decoder_output.shape}")

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    def reshape_fn(tensor1, tensor2):
        ten1, ten2 = tensor1
        ten1_reshaped = tf.reshape(ten1, (-1, ten1.shape[2]))
        ten2_reshaped = tf.reshape(ten2, (-1, ten2.shape[2]))
        ten3_reshaped = tf.reshape(tensor2, (-1, tensor2.shape[2]))
        return (ten1_reshaped,ten2_reshaped), ten3_reshaped

    # Assuming `dataset` is your PrefetchDataset
    train_dataset = train_dataset.map(reshape_fn)
    val_dataset = val_dataset.map(reshape_fn)


    # Debug: print shapes after batching
    print("Sample shapes after batching:")
    for (encoder_input, decoder_input), decoder_output in train_dataset.take(1):
        print(f"Encoder input shape: {encoder_input.shape}")
        print(f"Decoder input shape: {decoder_input.shape}")
        print(f"Decoder output shape: {decoder_output.shape}")

    # Create model
    vocab_size = min(len(tokenizer.word_index) + 1, args.vocab_size)
    model, encoder_model, decoder_model = create_seq2seq_model(
        vocab_size,
        args.max_transcript_length,
        args.max_title_length+1,
        embed_dim=args.embed_dim,
        lstm_units=args.lstm_units,
    )

    if not args.skip_training:
        print("Training model...")
        model, history, blue_scores = train_model(
            model, train_dataset, val_dataset, tokenizer,
            epochs=args.epochs, max_length=args.max_title_length
        )

        # Save the trained model
        model.save(args.model_file)

        # Evaluate the model
        print("\nEvaluating the model on the validation set:")
        bleu_score = evaluate_model(model, val_dataset, tokenizer, args.max_title_length)
        print(f"Final BLEU score: {bleu_score:.4f}")
    else:
        print("Skipping training as requested.")
        # Load existing model if skipping training
        model = tf.keras.models.load_model(args.model_file)

        # Evaluate the loaded model
        print("\nEvaluating the loaded model on the validation set:")
        bleu_score = evaluate_model(model, val_dataset, tokenizer, args.max_title_length)
        print(f"BLEU score: {bleu_score:.4f}")

    # Plot and save metrics
    plot_metrics(history, blue_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Title Prediction from Transcript - Training")
    parser.add_argument("--input_file", type=str, default="YT-titles-transcripts-clean.csv", help="Path to the input dataset file")
    parser.add_argument("--preprocessed_data_file", type=str, default="preprocessed_video_data.npz", help="Path to save preprocessed data")
    parser.add_argument("--max_transcript_length", type=int, default=2000, help="Maximum length of video transcript")
    parser.add_argument("--max_title_length", type=int, default=20, help="Maximum length of video title")
    parser.add_argument("--model_file", type=str, default="seq2seq_model.keras", help="Path to save trained model")
    parser.add_argument("--encoder_model_file", type=str, default="encoder_model.keras", help="Path to save encoder model")
    parser.add_argument("--decoder_model_file", type=str, default="decoder_model.keras", help="Path to save decoder model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and use existing model")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to use from the dataset")
    parser.add_argument("--vocab_size", type=int, default=500, help="Maximum vocabulary size")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--lstm_units", type=int, default=256, help="Number of LSTM units")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")

    args = parser.parse_args()
    main(args)
