import re
import pandas as pd
import torch
import argparse
from typing import List, Dict, Any, Optional
from datasets import load_dataset, concatenate_datasets
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def setup_nltk():
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

DEFAULT_DATA_FILES = [
    "hf://datasets/O1-OPEN/OpenO1-SFT/OpenO1-SFT.jsonl",
    "hf://datasets/O1-OPEN/OpenO1-SFT/OpenO1-SFT-Pro.jsonl"
]

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

def load_and_prepare(data_files: List[str]):
    datasets = []
    for file in data_files:
        dataset = load_dataset("json", data_files=file, split="train")
        if 'instruction' in dataset.column_names and 'output' in dataset.column_names:
            dataset = dataset.rename_columns({"instruction": "prompt", "output": "response"})
        datasets.append(dataset)
    return concatenate_datasets(datasets)

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
def process_batch(batch, batch_idx, translator, file_handle=None):
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
                if file_handle:
                    file_handle.write(f'\n"{translated_prompt}","{translated_pensamiento}","{translated_salida}","{prompt}","{response}"\n')
                    file_handle.flush()
        except Exception as e:
            print(f"Error in {i}: {e}")
            continue
            #return translated_batch
    print(f"Procesado lote {batch_idx}: {len(translated_batch['prompt'])} traducciones realizadas.")
    return translated_batch

def run(data_files: List[str],
        model_name: str,
        device: str,
        max_samples: Optional[int],
        interim_file: str,
        final_csv: str) -> None:
    setup_nltk()
    dev = device
    if device == 'auto':
        dev = 0 if torch.cuda.is_available() else -1
    translator = pipeline("translation", model=model_name, device=dev)
    unified_dataset = load_and_prepare(data_files)
    file_salida = open(interim_file, "w", encoding="utf-8")
    processed_dataset = process_batch(unified_dataset if max_samples is None else unified_dataset.select(range(min(len(unified_dataset), max_samples))), 1, translator, file_handle=file_salida)
    file_salida.close()
    df = pd.DataFrame(processed_dataset)
    df.to_csv(final_csv, index=False, encoding='utf-8')
    print(f"Traducción completada y guardada en '{final_csv}'")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Translate OpenO1-SFT datasets prompts and tagged segments to Spanish.")
    p.add_argument('--data-files', nargs='+', default=DEFAULT_DATA_FILES, help='List of JSONL HF paths or local files')
    p.add_argument('--model', default='Helsinki-NLP/opus-mt-en-es', help='Translation model')
    p.add_argument('--device', default='auto', help='Device: auto|-1|cpu|cuda index (e.g., 0)')
    p.add_argument('--max-samples', type=int, default=None, help='Limit number of samples (debug)')
    p.add_argument('--interim-file', default='coto1.csv', help='Interim write file for streaming progress')
    p.add_argument('--final-csv', default='translated_dataset_cot.csv', help='Final aggregated CSV output')
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(data_files=args.data_files,
        model_name=args.model,
        device=args.device,
        max_samples=args.max_samples,
        interim_file=args.interim_file,
        final_csv=args.final_csv)


if __name__ == '__main__':
    main()