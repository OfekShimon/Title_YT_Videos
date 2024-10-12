import matplotlib.pyplot as plt

def plot_metrics(history, bleu_scores):
    # Plot training & validation loss values
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Plot BLEU scores
    plt.subplot(132)
    plt.plot(bleu_scores)
    plt.title('BLEU Score')
    plt.ylabel('BLEU')
    plt.xlabel('Epoch')

    # If you have accuracy metrics, plot them here
    if 'accuracy' in history.history:
        plt.subplot(133)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    plt.close()

if __name__ == "__main__":
    print("This script is not meant to be run directly.")