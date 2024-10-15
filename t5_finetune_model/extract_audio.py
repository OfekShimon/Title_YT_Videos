import ffmpeg
import yt_dlp


def extract_audio(video, output_path):
    stream = ffmpeg.input(video)
    stream = ffmpeg.output(stream, output_path)
    ffmpeg.run(stream, overwrite_output=True)
    return output_path


def extract_youtube_audio(link, data_folder):
    print("downloading audio...")
    ydl_opts = {
        'extract_audio': True,
        'format': 'bestaudio',
        'outtmpl': data_folder + '%(title)s.wav'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(link, download=True)
        video_title = info_dict['title']
        ydl.download(link)
    print("finished downloading")

    audio_path = data_folder + video_title + '.wav'  # file name of your downloaded audio
    print("audio saved to: ", audio_path)

    return audio_path
