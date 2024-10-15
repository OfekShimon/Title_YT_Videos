import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import asyncio
import aiohttp
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


async def fetch_video_details(session, vid_id):
    try:
        return await yt8m_crawler.fetch_video_details_async(session, vid_id)
    except Exception as e:
        print(f"Error fetching video details for {vid_id}: {str(e)}")
        return None


async def process_example(session, example, genres):
    tf_example = tf.train.SequenceExample.FromString(example.numpy())

    labels = set(tf_example.context.feature['labels'].int64_list.value)
    if genres is not None and len(labels.intersection(genres)) == 0:
        return None

    yt8m_id = tf_example.context.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
    vid_id = await yt8m_crawler.get_real_id_async(session, yt8m_id)

    if vid_id is None:
        print(f'Error fetching video details: unable to get the real id of {yt8m_id}')
        return None

    n_frames = len(tf_example.feature_lists.feature_list['audio'].feature)
    rgb_frames = []
    audio_frames = []

    for i in range(n_frames):
        rgb_data = tf.io.decode_raw(tf_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],
                                    tf.uint8)
        rgb_frames.append(tf.cast(rgb_data, tf.float32).numpy().tolist())

        audio_data = tf.io.decode_raw(tf_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0],
                                      tf.uint8)
        audio_frames.append(tf.cast(audio_data, tf.float32).numpy().tolist())

    data_video = await fetch_video_details(session, vid_id)

    if data_video:
        return {
            'input': {
                'rgb': rgb_frames,
                'audio': audio_frames
            },
            'output': labels,
            'metadata': {
                'id': vid_id,
                'title': data_video.get('title', ''),
                'duration': data_video.get('duration', 0),
                'creator': data_video.get('uploader', ''),
                'views': data_video.get('view_count', 0),
                'likes': data_video.get('like_count', 0)
            }
        }
    return None


async def get_data_by_amount_async(data_amount=1000, type='train', output_path="data/merged_train_data.csv",
                                   metadata_filter=None, genres=None):
    data_type = "frame"
    base_url = "http://eu.data.yt8m.org/2"
    download_dir = "data/yt8m"

    tfrecord_index = 0
    input_output = []
    count = 0
    file_index = 1

    print(f"Starting data collection process for {data_amount} samples...")
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        while count < data_amount and tfrecord_index <= 3843:
            tfrecord_file = yt8m_downloader.download_tfrecord_by_index(base_url, download_dir, data_type, type,
                                                                       tfrecord_index)
            tfrecord_index += 1
            print(f"[{type}] Processing {tfrecord_file}")

            tasks = []
            for example in tf.data.TFRecordDataset(tfrecord_file):
                task = asyncio.create_task(process_example(session, example, genres))
                tasks.append(task)

            print(f"Processing {len(tasks)} examples from {tfrecord_file}")
            results = await asyncio.gather(*tasks)

            successful_results = [result for result in results if result is not None]
            print(f"Successfully processed {len(successful_results)} out of {len(tasks)} examples")

            for result in successful_results:
                if metadata_filter is None or metadata_filter(result['metadata']):
                    input_output.append(result)
                    count += 1
                    if count % (data_amount // 10) == 0:
                        elapsed_time = time.time() - start_time
                        print(
                            f"[{type}] Data preprocessing progress: {(count / data_amount) * 100:.2f}% ({count}/{data_amount})")
                        print(f"Elapsed time: {elapsed_time:.2f} seconds")
                        print(f"Estimated time remaining: {(elapsed_time / count) * (data_amount - count):.2f} seconds")
                    if count % 1000 == 0 or count == data_amount:
                        file_name = f'data/{type}_input_output_data_{count}.csv'
                        print(f"Saving intermediate results to {file_name}")
                        with open(file_name, 'w', newline='', encoding="utf-8") as csvfile:
                            fieldnames = ['rgb', 'audio', 'labels', 'id', 'title', 'creator', 'views', 'likes',
                                          'duration']
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            for item in input_output:
                                writer.writerow({
                                    'rgb': item['input']['rgb'],
                                    'audio': item['input']['audio'],
                                    'labels': item['output'],
                                    'id': item['metadata']['id'],
                                    'title': item['metadata']['title'],
                                    'creator': item['metadata']['creator'],
                                    'views': item['metadata']['views'],
                                    'likes': item['metadata']['likes'],
                                    'duration': item['metadata']['duration']
                                })
                        input_output = []
                        file_index += 1
                        print(f"Intermediate file {file_name} saved")
                if count == data_amount:
                    break

            if count == data_amount:
                break

    total_time = time.time() - start_time
    print(f"Data collection completed. Processed and saved {count} samples")
    print(f"Total execution time: {total_time:.2f} seconds")

    print(f"Merging intermediate files into {output_path}")
    input_pattern = f"data/{type}_input_output_data_*.csv"
    merge_csv_files(input_pattern, output_path)


def get_data_by_amount(data_amount=1000, type='train', output_path="data/merged_train_data.csv", metadata_filter=None,
                       genres=None):
    asyncio.run(get_data_by_amount_async(data_amount, type, output_path, metadata_filter, genres))


def load_and_prepare_data(csv_file_path='data/train_input_output_data_1000.csv', max_title_length=20, max_frames=300):
    print(f"Loading and preparing data from {csv_file_path}")
    # Load the CSV file
    data = pd.read_csv(csv_file_path)

    # Prepare X (input features)
    X_rgb = []
    X_audio = []
    titles = []

    print("Processing RGB and audio frames...")
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

    print("Preparing output labels (titles)...")
    # Prepare y (output labels)
    # Tokenize the titles
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(titles)
    y = tokenizer.texts_to_sequences(titles)

    # Pad the sequences
    y = pad_sequences(y, maxlen=max_title_length, padding='post')

    # Convert y to numpy array
    y = np.array(y)

    print("Data preparation completed")
    return X_rgb, X_audio, y, tokenizer


def decode_title(encoded_title, tokenizer):
    return ' '.join([tokenizer.index_word.get(idx, '') for idx in encoded_title if idx != 0])


def merge_csv_files(input_pattern, output_file):
    print(f"Merging CSV files matching pattern: {input_pattern}")
    # Get all CSV files matching the input pattern
    all_files = glob.glob(input_pattern)

    # Sort the files to ensure they are processed in order
    all_files.sort()

    print(f"Found {len(all_files)} files to merge")

    # List to store individual dataframes
    df_list = []

    # Read each CSV file and append to the list
    for i, filename in enumerate(all_files, 1):
        print(f"Reading file {i}/{len(all_files)}: {filename}")
        df = pd.read_csv(filename, index_col=None, header=0)
        df_list.append(df)

    print("Concatenating all dataframes...")
    # Concatenate all dataframes in the list
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)

    print(f"Writing merged data to {output_file}")
    # Write the combined dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)

    print(f"Merged {len(all_files)} files into {output_file}")


def main():
    print("Starting main process...")
    data_amount = 10000
    output_path = "data/merged_train_data.csv"

    print(f"Collecting {data_amount} samples...")
    get_data_by_amount(data_amount, 'train', output_path)

    print(f"Loading and preparing data from {output_path}")
    X_rgb, X_audio, y, tokenizer = load_and_prepare_data(output_path)

    print("Data preparation completed. Summary:")
    print("Shape of X_rgb:", X_rgb.shape)
    print("Shape of X_audio:", X_audio.shape)
    print("Shape of y:", y.shape)
    print("Vocabulary size:", len(tokenizer.word_index) + 1)
    print("\nExample decoded title:")
    print(decode_title(y[0], tokenizer))

    print("Process completed successfully")


if __name__ == "__main__":
    main()