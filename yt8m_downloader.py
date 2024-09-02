import os

import urllib.request

# Function to download a file if it doesn't already exist
def download_tfrecord(url, download_dir):
    file_name = os.path.basename(url)
    file_path = os.path.join(download_dir, file_name)
    
    # Check if the file already exists
    # if os.path.exists(file_path):
    #     print(f"{file_name} already exists. Skipping download.")
    # else:
    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"{file_name} downloaded successfully.")


def int_to_index(num):
    # indices are between 00 to ZZ
    if num > 3843:
        num = 3843
    first_num = num % (10 + 26 + 26)
    second_num = int(num / (10 + 26 + 26))
    index = ''
        
    if second_num < 10:
        index += chr(ord('0') + second_num)
    elif second_num < 10 + 26:
        index += chr(ord('a') + second_num - 10)
    else:
        index += chr(ord('A') + second_num - 10 - 26)
    
    if first_num < 10:
        index += chr(ord('0') + first_num)
    elif first_num < 10 + 26:
        index += chr(ord('a') + first_num - 10)
    else:
        index += chr(ord('A') + first_num - 10 - 26)
    
    return index


def download_tfrecords(base_url, download_dir, data_type, type, amount):
    download_dir = download_dir + '/' + data_type + '/' + type
    # Create the directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    for i in range(amount):
        download_tfrecord(base_url + '/' + data_type + '/' + type + '/' + type + int_to_index(i) + '.tfrecord', download_dir)