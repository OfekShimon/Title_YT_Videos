import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from extract_audio import extract_youtube_audio, extract_audio
from extract_transcription import transcribe

yt_link = 'https://www.youtube.com/watch?v=dHy-qfkO54E' # minecraft
data_folder = 't5_finetune_model/transcription data/'
transcription_path = data_folder + 'transcribe.txt'

# audio_path = extract_audio('path to video file', data_folder + 'audio.wav')
audio_path = extract_youtube_audio(yt_link, data_folder)
transcribe(audio_path, transcription_path)

# Load the model
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

out_dir = "t5_finetune_model/models"
# Load trained model
model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(out_dir, 'trained-v1')).to('cuda')

# Example input text for inference
input_text = 'summarize: ' + open(transcription_path, "r").read()

# Perform inference
with torch.no_grad():
    tokenized_text = tokenizer(input_text, truncation=True, padding=True, return_tensors='pt')

    source_ids = tokenized_text['input_ids'].to('cuda', dtype=torch.long)
    source_mask = tokenized_text['attention_mask'].to('cuda', dtype=torch.long)

    generated_ids = model.generate(
        input_ids=source_ids,
        attention_mask=source_mask,
        max_length=512,
        num_beams=5,
        repetition_penalty=1,
        length_penalty=1,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

print("\noutput:\n" + pred)