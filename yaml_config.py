import yaml

def load_config(file_path):
	try:
		with open(file_path, 'r', encoding='utf-8') as f:
			config = yaml.safe_load(f)
		return config
	except FileNotFoundError:
		print(f"file not found")
		return None
	except yaml.YAMLError as e:
		print("yaml load error")
		return None
		
		