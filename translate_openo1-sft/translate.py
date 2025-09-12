import re
import pandas as pd
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re


# Verificar disponibilidad de CUDA
#device = 0 if torch.cuda.is_available() else -1
device = "cpu"

# Cargar los datos sin que las columnas inconsistentes causen errores
# Primero cargamos todos los archivos JSON uno por uno
data_files = [
    "hf://datasets/O1-OPEN/OpenO1-SFT/OpenO1-SFT.jsonl",
    "hf://datasets/O1-OPEN/OpenO1-SFT/OpenO1-SFT-Pro.jsonl"
]
nltk.download('punkt')#, quiet=True)
nltk.download('punkt_tab')

def extract_text_between_tags(text, tag):
    """
    Detecta y extrae texto que está entre tags específicos.

    Args:
        text (str): El texto que contiene los tags.
        tag (str): El nombre del tag (sin los signos < >).

    Returns:
        list: Lista de textos encontrados entre los tags.
    """
    pattern = fr'<{tag}>(.*?)</{tag}>'
    return re.findall(pattern, text, re.DOTALL)

datasets = []
for file in data_files:
    # Intentar cargar con diferentes configuraciones
    dataset = load_dataset("json", data_files=file, split="train")
    # Detectar columnas y renombrarlas para unificación
    if 'instruction' in dataset.column_names and 'output' in dataset.column_names:
        dataset = dataset.rename_columns({"instruction": "prompt", "output": "response"})
    datasets.append(dataset)

# Concatenar todos los datasets procesados en uno solo
unified_dataset = concatenate_datasets(datasets)

# Cargar el modelo de traducción con soporte CUDA
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device=device)

file_salida = open("coto1.csv", "w", encoding="utf-8")

# Función para detectar idioma (Inglés vs Chino)
def is_english(text):
    # Patrón para caracteres en inglés (incluye letras, números, y algunos caracteres comunes)
    english_pattern = r'[A-Za-z0-9!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]'
    # Patrón para caracteres chinos
    chinese_pattern = r'[\u4e00-\u9fff]'
    # Detectar si hay caracteres chinos en el texto
    contains_chinese = re.search(chinese_pattern, text)
    # Detectar si hay caracteres en inglés en el texto
    contains_english = re.search(english_pattern, text)
    # Retornar True si contiene inglés y no chino
    return bool(contains_english and not contains_chinese)

# Procesar en lotes para evitar sobrecarga de memoria
def process_batch(batch, batch_idx):
    try:
        prompts = batch['prompt']
        responses = batch['response']
        translated_batch = {'prompt': [], 'response_thoughts': [], 'response_salida': [], 'original_prompt': [], 'original_response': []}
    except:
        return
    i = 0
    try:
        zipped = zip(prompts, responses)
    except:
        return
    for prompt, response in zipped:
        i += 1
        try:
            if is_english(prompt):
                # Traducir si está en inglés
                translated_prompt = translator(prompt, max_length=512)[0]['translation_text']
                thoughts = extract_text_between_tags(response, "Thought")
                pensamientos = []
                print("pensamientos: ", len(thoughts))
                for thought in thoughts:
                    sentences = sent_tokenize(thought)
                    print("oraciones: ", len(sentences))
                    for sent in sentences:
                        words = word_tokenize(sent)
                        refined_sentences = []
                        if len(words) > 470:
                            for i in range(0, len(words), 470):
                                refined_sentences.append(' '.join(words[i:i + 470]))
                            for rs in refined_sentences:
                                translated_sent = translator(rs, max_length=512)[0]['translation_text']
                                pensamientos.append(translated_sent)
                        else:
                            translated_sent = translator(sent, max_length=512)[0]['translation_text']
                            pensamientos.append(translated_sent)
                outputs = extract_text_between_tags(response, "Output")
                salidas = []
                for out in outputs:
                    print("traducir salida")
                    sentences = sent_tokenize(thought)
                    for sent in sentences:
                        translated_salida = translator(sent, max_length=512)[0]['translation_text']
                        salidas.append(translated_salida)
                #print(translated_prompt, "\n".join(pensamientos), salidas)
                # Agregar los resultados al lote traducido
                translated_pensamiento = " ".join(pensamientos)
                translated_salida = " ".join(salidas)
                translated_batch['prompt'].append(translated_prompt)
                translated_batch['response_thoughts'].append(translated_pensamiento)
                translated_batch['response_salida'].append(translated_salida)
                translated_batch['original_prompt'].append(prompt)
                translated_batch['original_response'].append(response)
                file_salida.write(f'\n"{translated_prompt}","{translated_pensamiento}","{translated_salida}","{prompt}","{response}"\n')
                file_salida.flush()
        except Exception as e:
            print(f"Error in {i}: {e}")
            continue
            #return translated_batch
    print(f"Procesado lote {batch_idx}: {len(translated_batch['prompt'])} traducciones realizadas.")
    return translated_batch

# Aplicar el procesamiento en el dataset usando `map` para procesamiento en lotes
#processed_dataset = unified_dataset.map(lambda batch, idx: process_batch(batch, idx), batched=True, batch_size=2, with_indices=True)
processed_dataset =process_batch(unified_dataset, 1)
file_salida.close()
# Guardar el resultado en un archivo CSV
df = pd.DataFrame(processed_dataset)
df.to_csv("translated_dataset_cot.csv", index=False, encoding='utf-8')

print("Traducción completada y guardada en 'translated_dataset_cot.csv'")