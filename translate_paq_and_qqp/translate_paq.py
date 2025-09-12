import os
import sys
from pathlib import Path

cache_directory = Path.cwd()

os.environ['HF_HOME'] = os.path.join(cache_directory, ".cache")
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(cache_directory, ".cache")
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_directory, ".cache")
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_directory, ".cache")



# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")


from datasets import load_dataset
dataset = load_dataset("embedding-data/PAQ_pairs", streaming=False, split="train") # stream for testing purposes

#print(dataset["train"][0]["set"][0])
#print(dataset["train"][0]["set"][1])


#sys.exit()


num_rows = 64371441
skip_n_rows = 6560
import pandas as pd
all_data = []
import pprint
i = -1
index = []
f_log = open("log.txt", "w")
from datetime import datetime

f_log.write("time,delta,item,event\n")
for data in dataset:
	i += 1
	if i < skip_n_rows:
		sys.exit()
	start_time = datetime.now()
	try:
		Q_original = data["set"][0]
		A_original = data["set"][1]
		log_msg = f"{datetime.now()},{datetime.now() - start_time},{i},A procesar\n"
		start_time = datetime.now()
		f_log.write(log_msg)
		print(log_msg)
		Q_traducida = pipe(data["set"][0])[0]['translation_text']
		A_traducida = pipe(data["set"][1])[0]['translation_text']
		d = {
			"Q_original": Q_original,
			"A_original": A_original,
			"Q_traducida": Q_traducida,
			"A_traducida": A_traducida
		}
		index.append(i)
		#pprint.pprint(d)
		log_msg = f"{datetime.now()},{datetime.now() - start_time},{i},Procesado\n"
		start_time = datetime.now()
		f_log.write(log_msg)
		print(log_msg)
		all_data.append(d)
		df = pd.DataFrame(all_data)
		df.index = index
		df.to_excel(excel_writer="dataset_paq_traducido.xlsx", sheet_name="Hoja1")

		log_msg = f"{datetime.now()},{datetime.now() - start_time},{i},Guardado\n"
		f_log.write(log_msg)
		print(log_msg)
	except Exception as e:
		log_msg = f"{datetime.now()},{datetime.now() - start_time},{i},Error: {e}\n"
		start_time = datetime.now()
		f_log.write(log_msg)
		print(log_msg)
	#if i >= 3:
log_msg = f"{datetime.now()},{datetime.now() - start_time},{i},Terminó\n"
start_time = datetime.now()
f_log.write(log_msg)
print(log_msg)
sys.exit()
