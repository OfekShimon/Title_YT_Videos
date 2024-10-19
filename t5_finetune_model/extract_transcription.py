# from faster_whisper import WhisperModel

# https://www.digitalocean.com/community/tutorials/how-to-generate-and-add-subtitles-to-videos-using-python-openai-whisper-and-ffmpeg

# def transcribe(audio_path, output_path):
#     model = WhisperModel("small", compute_type="int8")
#     print("Starting transcription")
#     segments, info = model.transcribe(audio_path)
#     # language = info[0]
#     print("Transcription language", info[0])
#     segments = list(segments)
#     # for segment in segments:
#     #     # print(segment)
#     #     print("[%.2fs -> %.2fs] %s" %
#     #           (segment.start, segment.end, segment.text))

#     with open(output_path, "w") as outfile:
#         outfile.write("\n".join([s.text for s in segments]))
#     print("Finished transcription")
    
from multiprocessing import Process
from faster_whisper import WhisperModel

def transcribe_impl(audio_path, output_path):
    model = WhisperModel("small", compute_type="int8")
    print("Starting transcription")
    segments, info = model.transcribe(audio_path)
    # language = info[0]
    print("Transcription language", info[0])
    segments = list(segments)
    # for segment in segments:
    #     # print(segment)
    #     print("[%.2fs -> %.2fs] %s" %
    #           (segment.start, segment.end, segment.text))

    with open(output_path, "w") as outfile:
        outfile.write("\n".join([s.text for s in segments]))
    print("Finished transcription")

# workaround to deal with a termination issue: https://github.com/guillaumekln/faster-whisper/issues/71
def transcribe(audio_path, output_path):
    p = Process(target=transcribe_impl, args=[audio_path, output_path])
    # p.daemon = True
    p.start()
    p.join()
    p.close()
