

async def get_substrings(s: str):
	substrings = []
	end = False
	while s != "":
		start = s.find("<")
		if end is not False:
			ss = s[:start]
			if ss != "" and len(ss.strip()) > 0:
				substrings.append((0, ss.strip()))
		end = s.find(">")
		substrings.append((1, s[start:end+1]))
		s = s[end+1:]
	return substrings

def is_int(s: str):
	try:
		int(s)
		return True
	except:
		return False


async def translate_substrings(substrings: str, pipe):
	new_substrings = []
	for type_, text in substrings:
		#print(f"{type_}\t{text}")
		if type_ == 0 and not is_int(text):
			translated = pipe(text)[0]['translation_text']
			new_substrings.append(translated)
			#print()
		else:
			new_substrings.append(text)
	return new_substrings



async def main():
	# Use a pipeline as a high-level helper
	from transformers import pipeline

	pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

	from datasets import load_dataset

	ds = load_dataset("qq8933/OpenLongCoT-Pretrain")

	import csv
	outputf = open("openlong_cot_es.csv", "w", encoding="utf-8")
	writerout = csv.writer(outputf, delimiter=",")
	i = 0
	for s in ds["train"]["text"]:
		try:
			new_s = await translate_substrings(await get_substrings(s), pipe)
		except Exception as e:
			print("ERROR: ", e)
			continue
		new_s = "".join(new_s)
		writerout.writerow([new_s])
		outputf.flush()
		print(i)
		i += 1
		#print(new_s)
		#break
	outputf.flush()
	outputf.close()

import asyncio
asyncio.run(main())