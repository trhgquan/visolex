import os
import numpy as np
import torch
import pandas as pd
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
from framework_components.student import Student
from utils import post_process, delete_special_tokens, get_tokenizer
from project_variables import DICT_PATH
from arguments import parse_arguments
from chatgpt import run_chatgpt
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()


def parse_html_to_string(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    cleaned_text = "\n".join(
        [line for line in text.splitlines() if line.strip()])
    return cleaned_text


def load_nsw_dict():
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    return dictionary


def add_vocab(nsw, response, dictionary):
    dictionary[nsw] = {}
    dictionary[nsw]['response'] = response

    with open(DICT_PATH, 'w') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)


def dict_lookup(nsw):
    nsw_dict = load_nsw_dict()
    if nsw in nsw_dict:
        normalized_data = nsw_dict[nsw]
        result = normalized_data
        try:
            normalized_data = result['response']['normalized'][0]
            response = {
                'word': normalized_data['word'],
                'definition': normalized_data['definition'],
                'abbreviation': normalized_data.get('abbreviations', ''),
                'example': normalized_data.get('example', '')
            }
            response_str = ""
            for key, value in response.items():
                string = "- " + key.upper() + ": " + value + "\n"
                response_str += string
        except:
            response_str = json.dumps(result)
    else:
        response = run_chatgpt(nsw)
        response = response.replace("```html\n", "").replace("\n```", "")
        add_vocab(nsw, response, nsw_dict)
        response_str = parse_html_to_string(response)

    return response_str


def concatenate_nsw_spans(nsw_spans):
    result = []

    # In case nsw_spans is blank
    if len(nsw_spans) > 0:
        current_span = nsw_spans[0]

        for i in range(1, len(nsw_spans)):
            next_span = nsw_spans[i]
            if current_span['end_index'] == next_span['start_index']:
                current_span['nsw'] += next_span['nsw']
                current_span['end_index'] = next_span['end_index']
            else:
                result.append(current_span)
                current_span = next_span
        result.append(current_span)

    return result


def nsw_detection(source_tokens, is_nsw, tokenizer):
    source_tokens, keep_indices = delete_special_tokens(source_tokens)
    is_nsw = [is_nsw[i] for i in keep_indices]
    nsw_indices = [i for i, nsw in enumerate(is_nsw) if nsw == 1]
    nsw_tokens = [source_tokens[i] for i in nsw_indices]

    nsw_spans = []
    end_index = 0
    for i in range(len(source_tokens)):
        if source_tokens[i].startswith('▁'):
            end_index += 1
        current_text = tokenizer.convert_tokens_to_string([source_tokens[i]])
        full_text = tokenizer.convert_tokens_to_string(source_tokens[:(i+1)])
        if is_nsw[i] == 1:
            if current_text:
                nsw_spans.append({
                    'index': i,
                    'start_index': end_index,
                    'end_index': end_index + len(current_text),
                    'nsw': current_text
                })
        end_index = len(full_text) if current_text else len(full_text) + 1

    # nsw_spans = concatenate_nsw_spans(nsw_spans)
    return nsw_spans


def lexnorm(output, tokenizer):

    # NSW Detection
    nsw_spans = nsw_detection(
        output['source_tokens'], output['is_nsw'], tokenizer)
    nsw_indices = [span['index'] for span in nsw_spans]

    # Lexical Normalization
    # pred = [id for id in output['pred'] if id != -1]
    # proba = [output['proba'][i] for i, id in enumerate(output['pred']) if id != -1]
    pred = output['pred']
    proba = output['proba']
    decoded_pred = tokenizer.convert_ids_to_tokens(pred)
    for i, nsw_idx in enumerate(nsw_indices):
        nsw_spans[i]['prediction'] = tokenizer.convert_tokens_to_string(
            [decoded_pred[nsw_idx+1]])
        nsw_spans[i]['confidence_score'] = round(proba[nsw_idx+1], 4)
    pred_tokens, keep_indices = delete_special_tokens(decoded_pred)
    proba = [proba[i] for i in keep_indices]
    pred_str = tokenizer.convert_tokens_to_string(pred_tokens)
    pred_str = post_process(pred_str)
    return nsw_spans, pred_str


def demo():
    args = parse_arguments()
    np.random.seed(args.seed)

    with open("config.json", "r+", encoding="utf-8") as f:
        inference_configs = json.load(f)

    TEXT_COL = inference_configs["text_col"]

    args.student_name = inference_configs["student_name"]
    args.rm_accent_ratio = 1.0 if inference_configs["remove_accents"] == "yes" else 0.0

    if args.rm_accent_ratio != 0.0:
        args.remove_accents = True

    # Load dataset
    df = pd.read_csv(inference_configs["dataset_path"])

    # Start Experiment
    args.logdir = os.path.join(args.experiment_folder, args.student_name,
                               args.training_mode + '_accent_{}'.format(str(args.rm_accent_ratio)))
    args.lower_case = True
    args.hard_student_rule = True
    args.soft_labels = True
    args.append_n_mask = True
    args.nsw_detect = True

    # Setup CUDA, GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    tokenizer = get_tokenizer(args.student_name)
    student = Student(args, tokenizer=tokenizer, logger=None)
    student.load("student_best")

    cleaned_res = []
    text_to_clean = df[TEXT_COL]
    for input in tqdm(text_to_clean):
        output = student.inference(user_input=input)
        _, pred_str = lexnorm(output, tokenizer=tokenizer)
        cleaned_res.append(pred_str)

    df[f"{args.student_name}_cleaned"] = cleaned_res

    df.to_csv(inference_configs["output_file"], index=False)


demo()
