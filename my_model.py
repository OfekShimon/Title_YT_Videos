import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from get_data_new import load_and_prepare_data, get_data_by_amount


class VideoTitleGenerator(nn.Module):
    def __init__(self, rgb_features, audio_features, vocab_size, max_title_length, embedding_dim=256, hidden_size=512):
        super(VideoTitleGenerator, self).__init__()
        self.rgb_features = rgb_features
        self.audio_features = audio_features
        self.vocab_size = vocab_size
        self.max_title_length = max_title_length
        self.hidden_size = hidden_size

        # Merge RGB and audio features
        self.feature_merger = nn.Sequential(
            nn.Linear(rgb_features + audio_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Encoder
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True, dropout=0.5)

        # Decoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers=2, batch_first=True,
                                    dropout=0.5)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, rgb_input, audio_input, decoder_input):
        # Merge RGB and audio features
        merged_features = torch.cat((rgb_input, audio_input), dim=-1)
        merged_features = self.feature_merger(merged_features)

        # Encoder
        _, (hidden, cell) = self.encoder_lstm(merged_features)

        # Decoder
        decoder_embedded = self.embedding(decoder_input)

        # Concatenate embedded input with context vector at each step
        context = hidden[-1].unsqueeze(1).repeat(1, decoder_embedded.size(1), 1)
        decoder_input = torch.cat((decoder_embedded, context), dim=2)

        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        output = self.output_layer(decoder_output)

        return output


def create_model(max_frames, rgb_features, audio_features, vocab_size, max_title_length, embedding_dim=256):
    model = VideoTitleGenerator(rgb_features, audio_features, vocab_size, max_title_length, embedding_dim)

    # Initialize weights with Xavier/Glorot initialization
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model_path, max_frames, rgb_features, audio_features, vocab_size, max_title_length):
    model = create_model(max_frames, rgb_features, audio_features, vocab_size, max_title_length)
    model.load_state_dict(torch.load(model_path))
    return model


def train_model(X_rgb, X_audio, y, tokenizer, max_title_length, epochs=100, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model(X_rgb.shape[1], X_rgb.shape[2], X_audio.shape[2], len(tokenizer.word_index) + 1,
                         max_title_length)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Prepare data
    X_rgb_tensor = torch.FloatTensor(X_rgb).to(device)
    X_audio_tensor = torch.FloatTensor(X_audio).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    dataset = TensorDataset(X_rgb_tensor, X_audio_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (batch_rgb, batch_audio, batch_y) in progress_bar:
            optimizer.zero_grad()

            # Teacher forcing with decreasing probability
            teacher_forcing_ratio = max(0.5, 1 - epoch / epochs)

            # Prepare decoder input (with start token)
            decoder_input = torch.full((batch_y.size(0), 1), tokenizer.word_index['<start>'], dtype=torch.long,
                                       device=device)

            loss = 0
            outputs = []
            for t in range(1, max_title_length):
                # Forward pass
                output = model(batch_rgb, batch_audio, decoder_input)
                outputs.append(output[:, -1:, :])

                # Teacher forcing
                if random.random() < teacher_forcing_ratio:
                    next_input = batch_y[:, t].unsqueeze(1)
                else:
                    next_input = output[:, -1, :].argmax(dim=-1).unsqueeze(1)

                decoder_input = torch.cat([decoder_input, next_input], dim=1)

            outputs = torch.cat(outputs, dim=1)
            loss = criterion(outputs.view(-1, outputs.size(-1)), batch_y[:, 1:].reshape(-1))

            if not torch.isfinite(loss):
                print(f"Loss is {loss.item()}, stopping training")
                return model, history

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predictions = outputs.argmax(dim=-1)
            correct_predictions += (predictions == batch_y[:, 1:]).sum().item()
            total_predictions += batch_y[:, 1:].ne(0).sum().item()

            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix({
                'batch': f'{batch_idx + 1}/{len(dataloader)}',
                'loss': f'{current_loss:.4f}',
                'accuracy': f'{current_accuracy:.4f}',
            })

        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_predictions
        epoch_time = time.time() - epoch_start_time

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Loss: {epoch_loss:.4f} (range: 0 - inf, lower is better)")
        print(f"Accuracy: {epoch_accuracy:.4f} (range: 0 - 1, higher is better)")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(epoch_loss)

        # Save best model
        if epoch == 0 or epoch_loss < min(history['loss'][:-1]):
            save_model(model, 'models/best_model.pth')

        # Print prediction examples
        if epoch % 10 == 0:
            print_predictions(model, X_rgb_tensor[:10], X_audio_tensor[:10], y_tensor[:10], tokenizer, max_title_length)

    # Save final model
    save_model(model, 'models/final_model.pth')

    return model, history
def print_predictions(model, X_rgb, X_audio, y, tokenizer, max_title_length):
    model.eval()
    with torch.no_grad():
        for i in range(10):
            decoder_input = torch.zeros((1, max_title_length), dtype=torch.long).to(X_rgb.device)
            decoder_input[0, 0] = tokenizer.word_index['<start>']

            for j in range(1, max_title_length):
                output = model(X_rgb[i:i + 1], X_audio[i:i + 1], decoder_input)
                predicted_token = output[0, j - 1].argmax().item()
                decoder_input[0, j] = predicted_token
                if predicted_token == tokenizer.word_index.get('<end>', 1):
                    break

            predicted_words = [tokenizer.index_word.get(idx.item(), '') for idx in decoder_input[0] if idx.item() != 0]
            predicted_title = ' '.join(predicted_words[1:-1])  # Remove start and end tokens
            actual_words = [tokenizer.index_word.get(idx.item(), '') for idx in y[i] if idx.item() != 0]
            actual_title = ' '.join(actual_words)

            print(f"Sample {i + 1}")
            print(f"Predicted: {predicted_title}")
            print(f"Actual: {actual_title}\n")


def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(122)
    plt.plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def predict_and_analyze(model, X_rgb, X_audio, y, tokenizer, max_title_length):
    model.eval()
    with torch.no_grad():
        decoder_input = torch.zeros((1, max_title_length), dtype=torch.long).to(X_rgb.device)
        decoder_input[0, 0] = tokenizer.word_index['<start>']

        for i in range(1, max_title_length):
            output = model(X_rgb, X_audio, decoder_input)
            predicted_token = output[0, i - 1].argmax().item()
            decoder_input[0, i] = predicted_token
            if predicted_token == tokenizer.word_index.get('<end>', 1):
                break

        predicted_words = [tokenizer.index_word.get(idx.item(), '') for idx in decoder_input[0] if idx.item() != 0]
        predicted_title = ' '.join(predicted_words[1:-1])  # Remove start and end tokens

        print("Raw prediction shape:", output.shape)
        print("Raw prediction sample (first 5 time steps, first 10 vocab):")
        print(output[0, :5, :10])

        print("Predicted word indices:", decoder_input[0])

        actual_words = [tokenizer.index_word.get(idx.item(), '') for idx in y[0] if idx.item() != 0]
        actual_title = ' '.join(actual_words)

        print("Predicted title:", predicted_title)
        print("Actual title:", actual_title)

        # Analyze prediction distribution
        plt.figure(figsize=(10, 5))
        plt.imshow(output[0].cpu().numpy().T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Probability')
        plt.title('Prediction Probabilities for Each Word')
        plt.xlabel('Time Step')
        plt.ylabel('Vocabulary Index')
        plt.tight_layout()
        plt.show()


def main():
    # Define hyperparameters
    data_amount = 10000
    batch_size = 64
    epochs = 100
    embedding_dim = 256

    # Load and prepare data
    train_data_path = "merged_train_data.csv"
    if not os.path.exists(train_data_path):
        print("Preprocessing data...")
        get_data_by_amount(data_amount, 'train', train_data_path)
    X_rgb, X_audio, y, tokenizer = load_and_prepare_data(train_data_path)

    print("X_rgb shape:", X_rgb.shape)
    print("X_audio shape:", X_audio.shape)
    print("y shape:", y.shape)
    print("Vocabulary size:", len(tokenizer.word_index) + 1)

    # Ensure '<start>' and '<end>' tokens are in the vocabulary
    tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1
    tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1
    tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}

    # Add start and end tokens to y
    y_with_tokens = np.zeros((y.shape[0], y.shape[1] + 2))
    y_with_tokens[:, 1:-1] = y
    y_with_tokens[:, 0] = tokenizer.word_index['<start>']
    y_with_tokens[:, -1] = tokenizer.word_index['<end>']
    y = y_with_tokens

    max_title_length = y.shape[1]

    # Convert data to PyTorch tensors
    X_rgb = torch.FloatTensor(X_rgb)
    X_audio = torch.FloatTensor(X_audio)
    y = torch.LongTensor(y)

    # Split data into train and validation sets
    X_rgb_train, X_rgb_val, X_audio_train, X_audio_val, y_train, y_val = train_test_split(
        X_rgb, X_audio, y, test_size=0.2, random_state=42)

    # Create 'models' directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Train model
    model, history = train_model(X_rgb_train, X_audio_train, y_train, tokenizer, max_title_length, epochs=epochs,
                                 batch_size=batch_size)

    # Print model summary
    print(model)

    # Plot training history
    plot_training_history(history)

    # Make predictions on a test sample
    test_rgb = X_rgb_val[:1]
    test_audio = X_audio_val[:1]
    predict_and_analyze(model, test_rgb, test_audio, y_val[:1], tokenizer, max_title_length)

    # Evaluate the model on the entire validation set
    print("\nEvaluating model on validation set...")
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        val_loss = 0
        val_correct_predictions = 0
        val_total_predictions = 0

        for i in range(0, len(X_rgb_val), batch_size):
            batch_rgb = X_rgb_val[i:i + batch_size].to(device)
            batch_audio = X_audio_val[i:i + batch_size].to(device)
            batch_y = y_val[i:i + batch_size].to(device)

            decoder_input = torch.zeros_like(batch_y)
            decoder_input[:, 0] = tokenizer.word_index['<start>']

            outputs = model(batch_rgb, batch_audio, decoder_input)
            loss = criterion(outputs.view(-1, outputs.size(-1)), batch_y[:, 1:].reshape(-1))

            val_loss += loss.item()

            predictions = outputs.argmax(dim=-1)
            val_correct_predictions += (predictions == batch_y[:, 1:]).sum().item()
            val_total_predictions += batch_y[:, 1:].ne(0).sum().item()

        val_loss /= (len(X_rgb_val) // batch_size)
        val_accuracy = val_correct_predictions / val_total_predictions

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save the tokenizer
    import pickle
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved to models/tokenizer.pkl")

    print("\nTraining and evaluation complete!")


if __name__ == "__main__":
    main()