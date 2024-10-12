run_10000:
	rm preprocessed_video_data.npz preprocessed_video_data.npz_tokenizer.pkl
	python main_script.py --num_samples 10000 --batch_size 128 --epochs 50 --vocab_size 20000 --embed_dim 256 --lstm_units 256 --dropout_rate 0.1

run_1000:
	rm preprocessed_video_data.npz preprocessed_video_data.npz_tokenizer.pkl
	python main_script.py --num_samples 1000 --batch_size 128 --epochs 20 --vocab_size 20000 --embed_dim 256 --lstm_units 256 --dropout_rate 0.1

run_100:
	rm preprocessed_video_data.npz preprocessed_video_data.npz_tokenizer.pkl
	python main_script.py --num_samples 100 --batch_size 64 --epochs 3 --vocab_size 20000 --embed_dim 256 --lstm_units 256 --dropout_rate 0.1

clean:
	rm preprocessed_video_data.npz preprocessed_video_data.npz_tokenizer.pkl