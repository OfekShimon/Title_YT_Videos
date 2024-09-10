from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import yt8m_downloader
import yt8m_crawler
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'



def get_data_by_amount(tfrecords_amount=100):
    data_type = "frame"
    base_url = "http://eu.data.yt8m.org/2"
    download_dir = "data/yt8m"
    #
    # yt8m_downloader.download_tfrecords(base_url, download_dir, data_type, 'train', tfrecords_amount)
    # yt8m_downloader.download_tfrecords(base_url, download_dir, data_type, 'validate', tfrecords_amount)
    # yt8m_downloader.download_tfrecords(base_url, download_dir, data_type, 'test', tfrecords_amount)

    frame_lvl_record = "data/yt8m/frame/train/train00.tfrecord"
    folder_path = 'data/yt8m/frame/train'

    # Get a list of all TFRecord files in the folder
    tfrecord_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tfrecord')]

    #limit to 10 files, for testing purposes
    tfrecord_files = tfrecord_files[:10]

    # Loop through each TFRecord file
    for tfrecord_file in tfrecord_files:
        print(tfrecord_file)

        input_output = []

        # for example in tf.data.TFRecordDataset(frame_lvl_record).take(10):

        for example in tf.data.TFRecordDataset(tfrecord_file).take(10):
            tf_example = tf.train.SequenceExample.FromString(example.numpy())

            vid_id = tf_example.context.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
            labels = list(tf_example.context.feature['labels'].int64_list.value)  # Convert to list

            n_frames = len(tf_example.feature_lists.feature_list['audio'].feature)
            rgb_frames = []
            audio_frames = []

            for i in range(n_frames):
                rgb_data = tf.io.decode_raw(tf_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0], tf.uint8)
                rgb_frames.append(tf.cast(rgb_data, tf.float32).numpy().tolist())  # Convert to list

                audio_data = tf.io.decode_raw(tf_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0], tf.uint8)
                audio_frames.append(tf.cast(audio_data, tf.float32).numpy().tolist())  # Convert to list

            try:
                data_video = yt8m_crawler.fetch_yt8m_video_details(vid_id)
            except Exception as e:
                print(f"Error fetching video details for {vid_id}: {str(e)}")
                continue

            if data_video:
                input_output.append({
                    'input': {
                        'rgb': rgb_frames,
                        'audio': audio_frames
                    },
                    'output': labels,
                    'metadata': {
                        'id': yt8m_crawler.get_real_id(vid_id),
                        'title': data_video.get('title', ''),
                        'creator': data_video.get('uploader', ''),
                        'views': data_video.get('view_count', 0),
                        'likes': data_video.get('like_count', 0),
                        'duration': data_video.get('duration', 0)
                    }
                })

    # Save input_output data
    with open('input_output_data.pkl', 'wb') as f:
        pickle.dump(input_output, f)

    print(f"Processed and saved {len(input_output)} samples")
    return input_output



def load_and_prepare_data(pickle_file_path='input_output_data.pkl', max_title_length=20, max_frames=300):
    # Load the pickle file
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)

    # Prepare X (input features)
    X_rgb = []
    X_audio = []
    titles = []

    for item in data:
        rgb_frames = item['input']['rgb']
        audio_frames = item['input']['audio']

        # Pad or truncate the frames
        if len(rgb_frames) > max_frames:
            rgb_frames = rgb_frames[:max_frames]
            audio_frames = audio_frames[:max_frames]
        else:
            rgb_frames = rgb_frames + [np.zeros_like(rgb_frames[0])] * (max_frames - len(rgb_frames))
            audio_frames = audio_frames + [np.zeros_like(audio_frames[0])] * (max_frames - len(audio_frames))

        X_rgb.append(rgb_frames)
        X_audio.append(audio_frames)
        titles.append(item['metadata']['title'])

    # Convert lists to numpy arrays
    X_rgb = np.array(X_rgb)
    X_audio = np.array(X_audio)

    # Prepare y (output labels)
    # Tokenize the titles
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(titles)
    y = tokenizer.texts_to_sequences(titles)

    # Pad the sequences
    y = pad_sequences(y, maxlen=max_title_length, padding='post')

    # Convert y to numpy array
    y = np.array(y)

    return X_rgb, X_audio, y, tokenizer

# Example of how to get the original title from y
def decode_title(encoded_title, tokenizer):
    return ' '.join([tokenizer.index_word.get(idx, '') for idx in encoded_title if idx != 0])


def main():
# Usage
    get_data_by_amount(2)
    X_rgb, X_audio, y, tokenizer = load_and_prepare_data()

    print("Shape of X_rgb:", X_rgb.shape)
    print("Shape of X_audio:", X_audio.shape)
    print("Shape of y:", y.shape)
    print("Vocabulary size:", len(tokenizer.word_index) + 1)
    print("\nExample decoded title:")
    print(decode_title(y[0], tokenizer))
