import json
import csv
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s'
                    , datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler("read_json.log", mode='w', encoding='utf-8'),
                        logging.StreamHandler()
                    ])

file = open("es-extract.jsonl", "r", encoding="utf-8", errors="ignore")
outfile = open("es_words.csv", "w", encoding="utf-8", newline="")
writer = csv.writer(outfile)
writer.writerow(["word", "definition"])

outfile_forms = open("es_word_forms.csv", "w", encoding="utf-8", newline="")
writer_forms = csv.writer(outfile_forms)
writer_forms.writerow(["word", "original form"])
word_forms = set()

ilines = 0
for line in file.readlines():
    ilines += 1
    if ilines % 10000 == 0:
        logging.info("=" * 100)
        logging.info(f"Línea {ilines}")
        logging.info("=" * 100)
    data = json.loads(line)
    if not "word" in data:
        continue
    word = data["word"]
    definiciones = []
    if data["lang_code"] != "es":
        continue
    if data["pos"] not in ["adj", "verb", "noun"]:
        continue
    for sense in data["senses"]:
        if "form_of" in sense:
            for form in sense["form_of"]:
                if "word" in form:
                    word_forms.add((word, form["word"]))
        tags = None
        if "tags" in sense:
            tags = sense["tags"]
        if tags:
            tags = sense["tags"]
            if "archaic" in tags or "obsolete" in tags:
                continue
            if "form-of" in tags:
                continue
        if not "glosses" in sense:
            continue
        for gloss in sense["glosses"]:
            definiciones.append(gloss)
    if definiciones:
        if len(definiciones) > 1:
            definiciones = list(set(definiciones))
            #logging.info(f"Palabra: {word} tiene {len(definiciones)} definiciones.")
        for definicion in definiciones:
            writer.writerow([word, definicion])
        outfile.flush()
for form in word_forms:
    writer_forms.writerow(form)
outfile_forms.flush()
file.close()
outfile.close()
outfile_forms.close()
