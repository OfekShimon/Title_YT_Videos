import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import tensorflow as tf
import yt8m_downloader
import yt8m_crawler
import random
import time
import csv
import glob

# save csv files with movies data, max 1000 per file
def get_data_by_amount(data_amount=1000, type='train', output_path="data/merged_train_data.csv", metadata_filter=None, genres=None):
    data_type = "frame"
    base_url = "http://eu.data.yt8m.org/2"
    download_dir = "data/yt8m"
    
    tfrecord_index = 0
    input_output = []
    count = 0
    file_index = 1
    first_print = True

    # Loop through each TFRecord file
    while count < data_amount and tfrecord_index <= 3843: # 3843 is the last file
        # download and get the name of the current tfrecord
        tfrecord_file = yt8m_downloader.download_tfrecord_by_index(base_url, download_dir, data_type, type, tfrecord_index)
        tfrecord_index += 1
        print(f"[{type}] working on {tfrecord_file}")


        for example in tf.data.TFRecordDataset(tfrecord_file):
            sleep_time = random.uniform(0.15, 0.55) # random.randint(1, 5) # 
            time.sleep(sleep_time)

            if count % 10 == 0 and first_print:
                print(f"fetching examples {count}-{min(count + 10, data_amount)}...")
                first_print = False
            tf_example = tf.train.SequenceExample.FromString(example.numpy())
            
            labels = set(tf_example.context.feature['labels'].int64_list.value)  # Convert to list
            if genres is not None and len(labels.intersection(genres)) == 0:
                continue

            yt8m_id = tf_example.context.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
            try:
                vid_id = yt8m_crawler.get_real_id(yt8m_id)
            except Exception as e:
                print(f"Error fetching video details for {yt8m_id}: {str(e)}")
                continue
            
            if vid_id is None:
                print(f'Error fetching video details: unable to get the real id of {yt8m_id}')
                continue

            n_frames = len(tf_example.feature_lists.feature_list['audio'].feature)
            rgb_frames = []
            audio_frames = []

            for i in range(n_frames):
                rgb_data = tf.io.decode_raw(tf_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0], tf.uint8)
                rgb_frames.append(tf.cast(rgb_data, tf.float32).numpy().tolist())  # Convert to list

                audio_data = tf.io.decode_raw(tf_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0], tf.uint8)
                audio_frames.append(tf.cast(audio_data, tf.float32).numpy().tolist())  # Convert to list

            try:
                data_video = yt8m_crawler.fetch_video_details(vid_id)
            except Exception as e:
                # print(f"Error fetching video details for {vid_id}: {str(e)}")
                continue

            if data_video and (metadata_filter is None or metadata_filter(data_video)):
                # print(f'title: {data_video.get('title', '')}, views: {data_video.get('view_count', 0)}')
                input_output.append({
                    'input': {
                        'rgb': rgb_frames,
                        'audio': audio_frames
                    },
                    'output': labels,
                    'metadata': {
                        'id': vid_id,
                        'title': data_video.get('title', ''),
                        'duration': data_video.get('duration', 0)
                    }
                })
                first_print = True
                count += 1
                if count % (data_amount / 10) == 0:
                    print(f"[{type}] data preprocessing progress: {(count / data_amount) * 100}%")
                if count % 1000 == 0 or count == data_amount:
                    file_name = f'data/{type}_input_output_data_{count}.csv'
                    with open(file_name, 'w', newline='', encoding="utf-8") as csvfile:
                        fieldnames = ['rgb', 'audio', 'labels', 'id', 'title', 'creator', 'views', 'likes',
                                      'duration']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        writer.writeheader()
                        for item in input_output:
                            writer.writerow({
                                'rgb': item['input']['rgb'],
                                'creator': data_video.get('uploader', ''),
                                'views': data_video.get('view_count', 0),
                                'likes': data_video.get('like_count', 0),
                                'audio': item['input']['audio'],
                                'labels': item['output'],
                                'id': item['metadata']['id'],
                                'title': item['metadata']['title'],
                                'duration': item['metadata']['duration']
                            })
                    input_output = []  # Reset the list for the next batch
                    file_index += 1

    print(f"Processed and saved {count} samples")
    input_pattern = "train_input_output_data_*.csv"
    merge_csv_files(input_pattern, output_path)
    return input_output


#load data from path
def load_and_prepare_data(csv_file_path='data/train_input_output_data_1000.csv', max_title_length=20, max_frames=300):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)

    # Prepare X (input features)
    X_rgb = []
    X_audio = []
    titles = []

    for _, row in data.iterrows():
        # Assuming the CSV has columns 'rgb', 'audio', and 'title'
        rgb_frames = np.fromstring(row['rgb'][1:-1], sep=',').reshape(-1, 3)  # Adjust shape as needed
        audio_frames = np.fromstring(row['audio'][1:-1], sep=',').reshape(-1, 1)  # Adjust shape as needed

        # Pad or truncate the frames
        if len(rgb_frames) > max_frames:
            rgb_frames = rgb_frames[:max_frames]
            audio_frames = audio_frames[:max_frames]
        else:
            rgb_frames = np.pad(rgb_frames, ((0, max_frames - len(rgb_frames)), (0, 0)), mode='constant')
            audio_frames = np.pad(audio_frames, ((0, max_frames - len(audio_frames)), (0, 0)), mode='constant')

        X_rgb.append(rgb_frames)
        X_audio.append(audio_frames)
        titles.append(row['title'])

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


def merge_csv_files(input_pattern, output_file):
    # Get all CSV files matching the input pattern
    all_files = glob.glob(input_pattern)

    # Sort the files to ensure they are processed in order
    all_files.sort()

    # List to store individual dataframes
    df_list = []

    # Read each CSV file and append to the list
    i = 0
    for filename in all_files:
        i+=1
        print("i: ", i)
        df = pd.read_csv(filename, index_col=None, header=0)
        df_list.append(df)

    # Concatenate all dataframes in the list
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)

    # Write the combined dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)

    print(f"Merged {len(all_files)} files into {output_file}")


def main():
# Usage
#     data_amount = 10000
#     get_data_by_amount(data_amount, 'train')
#     X_rgb, X_audio, y, tokenizer = load_and_prepare_data()
#
#     print("Shape of X_rgb:", X_rgb.shape)
#     print("Shape of X_audio:", X_audio.shape)
#     print("Shape of y:", y.shape)
#     print("Vocabulary size:", len(tokenizer.word_index) + 1)
#     print("\nExample decoded title:")
#     print(decode_title(y[0], tokenizer))

    input_pattern = "train_input_output_data_*.csv"
    output_file = "merged_train_data.csv"
    merge_csv_files(input_pattern, output_file)


if __name__ == "__main__":
    main()