import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from extract_audio import extract_youtube_audio, extract_audio
from extract_transcription import transcribe

if __name__ == "__main__":
    link = 'https://www.youtube.com/watch?v=dHy-qfkO54E' # minecraft
    data_folder = 'transcription data/'
    audio_path = 'transcription data/audio.wav'
    transcription_path = 'transcription data/'

    if link.startswith('https://www.youtube.com/'):
        video_id = link.split('watch?v=')[-1]
        audio_transcription_path = transcription_path + f'transcription-{video_id}.txt'
        if not os.path.exists(audio_transcription_path):
            original_title, audio_path = extract_youtube_audio(link, 'transcription data/')
            transcribe(audio_path, audio_transcription_path)
    else:
        file_name = os.path.basename(link)
        audio_transcription_path = transcription_path + f'transcription-{file_name}.txt'
        if not os.path.exists(audio_transcription_path):
            extract_audio(link, audio_path)
            transcribe(audio_path, audio_transcription_path)

    token_max_length = 512
    out_dir = "t5_finetune_model/models"
    # Load trained model
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(out_dir, 'trained-v4')).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained("t5_finetune_model/models/trained-v4")
    tokenizer.model_max_length = token_max_length

    # Example input text for inference
    input_text = 'summarize: ' + open(audio_transcription_path, "r").read()

    # Perform inference
    with torch.no_grad():
        tokenized_text = tokenizer(input_text, truncation=True, padding=True, return_tensors='pt', max_length=token_max_length)

        source_ids = tokenized_text['input_ids'].to('cuda', dtype=torch.long)
        source_mask = tokenized_text['attention_mask'].to('cuda', dtype=torch.long)

        generated_ids = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_new_tokens=20,
            num_beams=5,
            repetition_penalty=1,
            length_penalty=1,
            early_stopping=True,
            no_repeat_ngram_size=2,
            num_return_sequences=5
        )

        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    print('5 generated title suggestions:')
    for pred in preds:
        print(pred)