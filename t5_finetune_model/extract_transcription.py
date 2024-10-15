from faster_whisper import WhisperModel

# https://www.digitalocean.com/community/tutorials/how-to-generate-and-add-subtitles-to-videos-using-python-openai-whisper-and-ffmpeg

def transcribe(audio_path, output_path):
    model = WhisperModel("small")
    print("Starting transcription")
    segments, info = model.transcribe(audio_path)
    language = info[0]
    print("Transcription language", info[0])
    segments = list(segments)
    # for segment in segments:
    #     # print(segment)
    #     print("[%.2fs -> %.2fs] %s" %
    #           (segment.start, segment.end, segment.text))

    with open(output_path, "w") as outfile:
        outfile.write("\n".join([s.text for s in segments]))
    print("Finished transcription")
    
    # return the language and segments in case they are wanted
    return language, segments