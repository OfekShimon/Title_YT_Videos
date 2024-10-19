import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter import filedialog
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from extract_audio import extract_youtube_audio, extract_audio
from extract_transcription import transcribe
import pandas as pd
import torch
import threading
import os
import time

class Application(ttk.Frame):
    def __init__(self, master=None):
        master.title("YouTube Video Title Generator")
        master.geometry("800x800")

        style = ttk.Style()
        style.configure('TButton', font=("Helvetica", 12))

        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.create_widgets()
        self.select_model()

    def create_widgets(self):
        self.link_label = ttk.Label(self, text="Enter a YouTube link or select a local video file", font=("Helvetica", 12))
        self.link_label.pack(side="top", padx=10, pady=10)

        self.link_entry = ttk.Entry(self, font=("Helvetica", 12), width=40)
        self.link_entry.pack(side="top", padx=10, pady=10)
        self.link_entry.insert(0, "https://www.youtube.com/watch?v=dHy-qfkO54E")

        self.local_video_button = ttk.Button(self, text="Select local video", command=self.select_local_video)
        self.local_video_button.pack(side="top", padx=10, pady=10)

        self.model_label = ttk.Label(self, text="Select a model", font=("Helvetica", 12))
        self.model_label.pack(side="top", padx=10, pady=10)

        self.model_var = tk.StringVar()
        if os.path.exists("t5_finetune_model/models/trained-v1"):
            self.model_var.set("t5-small v1")
        elif os.path.exists("t5_finetune_model/models/trained-v4"):
            self.model_var.set("t5-small v2")
        elif os.path.exists("t5_finetune_model/models/trained-base-v0"):
            self.model_var.set("t5-base v1")
        else:
            self.model_var.set("None")

        self.model_frame = ttk.Frame(self)
        self.model_frame.pack(side="top", padx=10, pady=10)

        self.model_v1 = ttk.Radiobutton(self.model_frame, text="t5-small v1", variable=self.model_var, value="t5-small v1")
        self.model_v1.pack(side="left", padx=10)

        self.model_v4 = ttk.Radiobutton(self.model_frame, text="t5-small v2", variable=self.model_var, value="t5-small v2")
        self.model_v4.pack(side="left", padx=10)

        self.model_base = ttk.Radiobutton(self.model_frame, text="t5-base v1", variable=self.model_var, value="t5-base v1")
        self.model_base.pack(side="left", padx=10)

        self.model_file_button = ttk.Radiobutton(self.model_frame, text="Browse model file", variable=self.model_var, value="browse")
        self.model_file_button.pack(side="left", padx=10)

        self.model_button = ttk.Button(self, text="Select model", command=self.select_model)
        self.model_button.pack(side="top", padx=10, pady=10)

        self.token_max_length_label = ttk.Label(self, text="Token max length", font=("Helvetica", 12))
        self.token_max_length_label.pack(side="top", padx=10, pady=10)

        self.token_max_length_entry = ttk.Entry(self, font=("Helvetica", 12), width=10)
        self.token_max_length_entry.pack(side="top", padx=10, pady=10)
        self.token_max_length_entry.insert(0, "512")
        self.token_max_length_entry.bind("<KeyRelease>", self.update_token_max_length)

        self.generate_button = ttk.Button(self, text="Generate title", command=self.generate_title)
        self.generate_button.pack(side="top", padx=10, pady=10)

        self.randomize_frame = ttk.Frame(self)
        self.randomize_frame.pack(side="top", padx=10, pady=10)

        self.randomize_button = ttk.Button(self.randomize_frame, text="Run model on 5 random videos", command=self.run_on_random_videos)
        self.randomize_button.pack(side="left", padx=10)

        self.refresh_button = ttk.Button(self.randomize_frame, text="Refresh", command=self.refresh_random_videos)
        self.refresh_button.pack(side="left", padx=10)

        self.title_text = tk.Text(self, font=("Helvetica", 12), width=40, height=10)
        self.title_text.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        self.title_text.configure(state='disabled')

        self.model = None
        self.tokenizer = None
        self.token_max_length = 512
        self.dataset = pd.read_csv("t5_finetune_model/YT-titles-transcripts-clean.csv")
        self.random_videos = self.dataset.sample(n=5, random_state=42)
    
    def refresh_random_videos(self):
        """Refresh the random videos"""
        self.random_videos = self.dataset.sample(n=5)
    
    def run_on_random_videos(self):
        """Run the model on 5 random videos from YT-titles-transcripts-clean.csv"""
        original_titles = self.random_videos["title"].tolist()
        generated_titles = []
        for transcript in self.random_videos["transcript"]:
            input_ids = self.tokenizer.encode_plus(transcript, 
                                                    add_special_tokens=True, 
                                                    max_length=self.token_max_length,
                                                    padding='max_length', 
                                                    truncation=True, 
                                                    return_attention_mask=True, 
                                                    return_tensors='pt')
            output = self.model.generate(input_ids['input_ids'].to('cuda'), 
                                        attention_mask=input_ids['attention_mask'].to('cuda'),
                                        max_new_tokens=20,
                                        num_beams=5,
                                        repetition_penalty=1,
                                        length_penalty=1,
                                        early_stopping=True,
                                        no_repeat_ngram_size=2,
                                        num_return_sequences=5)
            generated_titles.append(self.tokenizer.decode(output[0], skip_special_tokens=True))
        self.title_text.configure(state='normal')
        self.title_text.delete(1.0, tk.END)
        for i in range(5):
            self.title_text.insert(tk.END, f"Original title: {original_titles[i]}\n")
            self.title_text.insert(tk.END, f"Generated title: {generated_titles[i]}\n\n")
        self.title_text.configure(state='disabled')
    
    def select_local_video(self):
        path = filedialog.askopenfilename()
        self.link_entry.delete(0, tk.END)
        self.link_entry.insert(0, path)

    def select_model(self):
        if self.model_var.get() == "t5-small v1":
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5_finetune_model/models/trained-v1").to('cuda')
            self.tokenizer = AutoTokenizer.from_pretrained("t5_finetune_model/models/trained-v1")
        elif self.model_var.get() == "t5-small v2":
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5_finetune_model/models/trained-v4").to('cuda')
            self.tokenizer = AutoTokenizer.from_pretrained("t5_finetune_model/models/trained-v4")
        elif self.model_var.get() == "t5-base v1":
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5_finetune_model/models/trained-base-v0").to('cuda')
            self.tokenizer = AutoTokenizer.from_pretrained("t5_finetune_model/models/trained-base-v0")
        elif self.model_var.get() == "browse":
            path = filedialog.askdirectory()
            if path == "":
                self.model_var.set(self.previous_model)
                return
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to('cuda')
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.previous_model = self.model_var.get()
        self.model_label["text"] = "Model: " + self.model_var.get()

    def update_token_max_length(self, event):
        try:
            self.token_max_length = int(self.token_max_length_entry.get())
        except ValueError:
            self.token_max_length_entry.delete(0, tk.END)
            self.token_max_length_entry.insert(0, str(self.token_max_length))
        
    def calculate_title(self):
        audio_path = 't5_finetune_model/transcription data/audio.wav'
        transcription_path = 't5_finetune_model/transcription data/'

        if self.link_entry.get().startswith('https://www.youtube.com/'):
            video_id = self.link_entry.get().split('watch?v=')[-1]
            audio_transcription_path = transcription_path + f'transcription-{video_id}.txt'
            if not os.path.exists(audio_transcription_path):
                _, audio_path = extract_youtube_audio(self.link_entry.get(), 't5_finetune_model/transcription data/')
                self.title_text.configure(state='normal')
                self.title_text.delete(1.0, tk.END)
                self.title_text.insert(tk.END, "Transcribing audio...")
                self.title_text.update_idletasks()
                self.title_text.configure(state='disabled')
                transcribe(audio_path, audio_transcription_path)
        else:
            file_name = os.path.basename(self.link_entry.get())
            audio_transcription_path = transcription_path + f'transcription-{file_name}.txt'
            if not os.path.exists(audio_transcription_path):
                extract_audio(self.link_entry.get(), audio_path)
                self.title_text.configure(state='normal')
                self.title_text.delete(1.0, tk.END)
                self.title_text.insert(tk.END, "Transcribing audio...")
                self.title_text.update_idletasks()
                self.title_text.configure(state='disabled')
                transcribe(audio_path, audio_transcription_path)

        self.title_text.configure(state='normal')
        self.title_text.delete(1.0, tk.END)
        self.title_text.insert(tk.END, "Generating title...")
        self.title_text.update_idletasks()
        self.title_text.configure(state='disabled')

        input_text = 'summarize: ' + open(audio_transcription_path, "r").read()

        with torch.no_grad():
            tokenized_text = self.tokenizer(input_text, truncation=True, padding=True, return_tensors='pt', max_length=self.token_max_length)
            source_ids = tokenized_text['input_ids'].to('cuda', dtype=torch.long)
            source_mask = tokenized_text['attention_mask'].to('cuda', dtype=torch.long)
            generated_ids = self.model.generate(
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
            preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

        self.title_text.configure(state='normal')
        self.title_text.delete(1.0, tk.END)
        self.title_text.insert(tk.END, 'Generated titles:\n' + '\n'.join(preds))
        self.title_text.configure(state='disabled')

        self.link_entry["state"] = tk.NORMAL
        self.local_video_button["state"] = tk.NORMAL
        self.model_button["state"] = tk.NORMAL
        self.generate_button["state"] = tk.NORMAL
        self.token_max_length_entry["state"] = tk.NORMAL

    def generate_title(self):
        if self.model is None:
            self.title_text.configure(state='normal')
            self.title_text.delete(1.0, tk.END)
            self.title_text.insert(tk.END, "Select a model first")
            self.title_text.update_idletasks()
            self.title_text.configure(state='disabled')
            return
        self.title_text.configure(state='normal')
        self.title_text.delete(1.0, tk.END)
        self.title_text.insert(tk.END, "Extracting audio...")
        self.title_text.update()
        self.title_text.configure(state='disabled')

        self.link_entry["state"] = tk.DISABLED
        self.local_video_button["state"] = tk.DISABLED
        self.model_button["state"] = tk.DISABLED
        self.generate_button["state"] = tk.DISABLED
        self.token_max_length_entry["state"] = tk.DISABLED
            
        thread = threading.Thread(target=self.calculate_title)
        thread.start()

if __name__ == "__main__":
    root = ThemedTk(theme="Arc")
    app = Application(master=root)
    app.mainloop()





