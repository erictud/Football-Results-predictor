all: data analyze train

data:
	python3 extract_data.py

analyze: data
	python3 analyze_data.py

train: analyze
	python3 train_model.py