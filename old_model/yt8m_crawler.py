from urllib.request import Request, urlopen

# pip install yt-dlp
import yt_dlp

import requests
from bs4 import BeautifulSoup


# Function to map the yt8m id to the real YouTube video id
def get_real_id(yt8m_id):
    url = 'http://data.yt8m.org/2/j/i/{}/{}.js'.format(yt8m_id[0:2], yt8m_id)
    req = Request(
        url, 
        headers={'User-Agent': 'Mozilla/6.0'}
    )
    try:
        request = urlopen(req).read()
    except:
        return None
    real_id = request.decode()
    return real_id[real_id.find(',') + 2:real_id.find(')') - 1]


# # We need this function to filter out the fields of metadata we won't be using about each video
# def without_keys(d, wanted_data):
#     return {x: d[x] for x in d if x in wanted_data}


def fetch_video_details(video_id):
    """
    Fetches video details using the YouTube video URL.
    
    Returns:
    - dict: A dictionary containing video details - title and view count
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Send a GET request to the YouTube video URL
    response = requests.get(video_url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve YouTube page. Status code: {response.status_code}")
        return None
    
    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Get video title
    title = soup.find('meta', {'name': 'title'})['content']
    
    # Get video view count (It may vary depending on the page structure, this is one common method)
    views = soup.find('meta', {'itemprop': 'interactionCount'})['content']
    
    return {
        'title': title,
        'view_count': int(views)
    }

    """video_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            video_info = ydl.extract_info(video_url, download=False)
        except yt_dlp.utils.DownloadError as e:
            print(f"Error fetching video details: {e}")
            return None

    return video_info"""


def fetch_yt8m_video_details(yt8m_id):
    """
    Fetches video details using the YouTube video URL.
    
    Returns:
    - dict: A dictionary containing video details such as title, view count, and more.
    """
    return fetch_video_details(get_real_id(yt8m_id))