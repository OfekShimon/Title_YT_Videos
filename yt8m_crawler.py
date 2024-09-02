from urllib.request import urlopen

# pip install yt-dlp
import yt_dlp


# Function to map the yt8m id to the real YouTube video id
def get_real_id(yt8m_id):
    url = 'http://data.yt8m.org/2/j/i/{}/{}.js'.format(yt8m_id[0:2], yt8m_id)
    request = urlopen(url).read()
    real_id = request.decode()
    return real_id[real_id.find(',') + 2:real_id.find(')') - 1]


# # We need this function to filter out the fields of metadata we won't be using about each video
# def without_keys(d, wanted_data):
#     return {x: d[x] for x in d if x in wanted_data}


def fetch_video_details(video_id):
    """
    Fetches video details using the YouTube video URL.
    
    Returns:
    - dict: A dictionary containing video details such as title, view count, and more.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': 'best'
    }

    with  yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            video_info = ydl.extract_info(video_url, download=False)
        except yt_dlp.utils.DownloadError as e:
            print(f"Error fetching video details: {e}")
            return None

    return video_info


def fetch_yt8m_video_details(yt8m_id):
    """
    Fetches video details using the YouTube video URL.
    
    Returns:
    - dict: A dictionary containing video details such as title, view count, and more.
    """
    return fetch_video_details(get_real_id(yt8m_id))