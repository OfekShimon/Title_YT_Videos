import numpy as np
import pandas as pd

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import yt8m_downloader
import yt8m_crawler

# data type in the tfrecords: video/frame
# refrence: https://research.google.com/youtube8m/download.html
# Frame level tf records: http://eu.data.yt8m.org/2/frame/train/index.html          same with validate and text
# Video level tf records: http://eu.data.yt8m.org/2/video/train/index.html          same with validate and text
data_type = "frame"
# Where to download tfrecords from
base_url = "http://eu.data.yt8m.org/2"
# Directory to save the downloaded files
download_dir = "data/yt8m"

tfrecords_amount = 2
yt8m_downloader.download_tfrecords(base_url, download_dir, data_type, 'train', tfrecords_amount)
yt8m_downloader.download_tfrecords(base_url, download_dir, data_type, 'validate', tfrecords_amount)
yt8m_downloader.download_tfrecords(base_url, download_dir, data_type, 'test', tfrecords_amount)

# tf.config.list_physical_devices('GPU')


# Example usage:
if __name__ == "__main__":

    # The path to the TensorFlow record
    frame_lvl_record = "data/yt8m/frame/train/train00.tfrecord"
    
    vid_ids = []
    labels = []
    feat_rgb = []
    feat_audio = []
    rows = []

    # Iterate the contents of the TensorFlow record
    for example in tf.data.TFRecordDataset(frame_lvl_record).take(100):
        tf_example = tf.train.SequenceExample.FromString(example.numpy())
        
        # Once we have the structured data, we can extract the relevant features (id and labels)
        vid_ids.append(tf_example.context.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
        yt8m_id = vid_ids[-1]
        real_id = yt8m_crawler.get_real_id(yt8m_id)
        if real_id is None:
            print(f'Error fetching video details: unable to get the real id of {yt8m_id}')
            continue
        labels.append(tf_example.context.feature['labels'].int64_list.value)
        
        n_frames = len(tf_example.feature_lists.feature_list['audio'].feature)
        # sess = tf.InteractiveSession()
        rgb_frame = []
        audio_frame = []
        # iterate through frames
        for i in range(n_frames):
            # Decode the raw bytes and convert to float32
            rgb_data = tf.io.decode_raw(tf_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0], tf.uint8)
            rgb_frame.append(tf.cast(rgb_data, tf.float32).numpy())

            audio_data = tf.io.decode_raw(tf_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0], tf.uint8)
            audio_frame.append(tf.cast(audio_data, tf.float32).numpy())
        
        
        # sess.close()
        feat_rgb.append(rgb_frame)
        feat_audio.append(audio_frame)
        
        # Get the yt-dlp metadata
        data_video = yt8m_crawler.fetch_video_details(real_id)
        
        if data_video:
            # We are interested in expanding the labels information with features such as title, 
            # creator, views, likes and duration
            title = data_video['title']
            creator = data_video['uploader']
            views = data_video['view_count']
            likes = data_video['like_count']
            duration = data_video['duration']
            
            # Collect the data in the dataframe
            rows.append({'id': real_id, 
                                'title': title, 
                                'creator': creator, 
                                'views': views,
                                'likes': likes,
                                'duration': duration,
                                'labels': labels[-1]})
    

    data = pd.DataFrame(rows)
    #print(data.describe())
    data.to_csv('yt8m_data.csv')
    
    video_id = "dQw4w9WgXcQ"  # Example video ID
    video_details = yt8m_crawler.fetch_video_details(video_id)
    
    if video_details:
        print("Title:", video_details['title'])
        print("View Count:", video_details['view_count'])
        print("Duration (seconds):", video_details['duration'])
        print("Like Count:", video_details['like_count'])
        print("Upload Date:", video_details['upload_date'])
        # print("Description:", video_details['description'])