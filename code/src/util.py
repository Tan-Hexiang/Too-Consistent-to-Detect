# -*- coding: UTF-8 -*-
from asyncio.log import logger
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re
import regex
import string
import torch
import jsonlines
from filelock import FileLock


def init_llama_from_local(model_name, model_path):


    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if '70b' in model_name:
        print("fp16")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda")
    model.eval()
    return model, tokenizer

def init_llama(model_name, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if '70b' in model_name:
        print("fp16")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda")
    model.eval()

    return model, tokenizer

def extract_contrast_sentences(text):
    pattern = r'• (.*?)(?=\n|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def match_response(input_string):
    marker = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    marker_position = input_string.find(marker)
    if marker_position != -1:
        result = input_string[marker_position + len(marker):]
        eot_position = result.find("<|eot_id|>")
        if eot_position != -1:
            result = result[:eot_position]
        end_of_text_position = result.find("<|end_of_text|>")
        if end_of_text_position!=-1:
             result = result[:end_of_text_position]
        return result
    else:
        print("Pattern not found.")
        return input_string


# check exist
def check_exist(data, exist_data_path):
        if os.path.exists(exist_data_path):
            exist_data = load_all_jsonl(exist_data_path)
            flag = []
            for k, example in enumerate(data):
                if find_with_id(example['id'], exist_data) == None:
                    flag.append(k)
            new_data = []
            for i in flag:
                new_data.append(data[i])
            logging.info("Total data: {}, Exist data: {}, Data to proess: {}".format(len(data), len(exist_data), len(new_data)))
            print("Total data: {}, Exist data: {}, Data to proess: {}".format(len(data), len(exist_data), len(new_data)))
            return new_data        
        else:
            logging.info("Total data: {}, Exist data: {}, Data to proess: {}".format(len(data), 0, len(data)))
            print("Total data: {}, Exist data: {}, Data to proess: {}".format(len(data), 0, len(data)))
            return data


def find_with_id(id, data, answers=None):
    for item in data:
        if int(item['id']) == int(id):
            if answers == None:
                return item
            else:
                if answers == item['answer']:
                    return item
    return None

# ----- Metric ----- #

# EM correctness
def normalize_answer(s):
 def remove_articles(text):
  return regex.sub(r'\b(a|an|the)\b', ' ', text)

 def white_space_fix(text):
  return ' '.join(text.split())

 def remove_punc(text):
  exclude = set(string.punctuation)
  return ''.join(ch for ch in text if ch not in exclude)

 def lower(text):
  return text.lower()

 return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
 return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
 return max([exact_match_score(prediction, gt) for gt in ground_truths])

# ----- Common tools ----- #
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def config_log(dir, name):
    log_path = dir+"/{}.log".format(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(
        format='%(thread)d, %(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path, mode='w'),
                              stream_handler]
    )

def dump_all_jsonl(data, output_path):
    lock_path = output_path + '.lock'
    with FileLock(lock_path):
        with jsonlines.open(output_path, "a") as writer:
            for item in data:
                writer.write(item)

def dump_jsonl(data, output_path):
    lock_path = output_path + '.lock'
    with FileLock(lock_path):
        with jsonlines.open(output_path, "a") as writer:
            writer.write(data)

def load_all_jsonl(input_path) -> list:
    data = []
    with jsonlines.open(input_path) as reader:
        for item in reader:
            data.append(item)
    return data

def init_logger(log_file: str):
    logger = logging.getLogger()

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
        
    logger.setLevel(logging.INFO) 
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) 
    
    file_handler = logging.FileHandler(log_file, encoding='utf8')
    file_handler.setLevel(logging.INFO) 
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# metric
from netcal.metrics import ECE
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import numpy as np
def compute_conf_metrics(y_true, y_confs, verbalise = True):
    result_matrics = {}
    # ACC
    accuracy = sum(y_true) / len(y_true)
    result_matrics['acc'] = accuracy

    # assert all([x >= 0 and x <= 1 for x in y_confs]), y_confs
    y_confs, y_true = np.array(y_confs), np.array(y_true)
    # print(y_true)
    # AUCROC
    roc_auc = roc_auc_score(y_true, y_confs)
    result_matrics['auroc'] = roc_auc

    # ECE
    n_bins = 10
    # diagram = ReliabilityDiagram(n_bins)
    ece = ECE(n_bins)
    ece_score = ece.measure(np.array(y_confs), np.array(y_true))
    result_matrics['ece'] = ece_score
    
    # brier score
    brier_score = brier_score_loss(y_true, y_confs, pos_label=1)
    
    result_matrics['bs'] = brier_score
    if verbalise:
        print("accuracy: ", accuracy)
        print("ROC AUC score:", roc_auc)
        print("ECE:", ece_score)
        print("Brier Score:", brier_score)

    return result_matrics



def compute_auroc(y_true, y_confs):
    y_confs, y_true = np.array(y_confs), np.array(y_true)
    # AUCROC
    roc_auc = roc_auc_score(y_true, y_confs)
    return roc_auc

def compute_accuracy(label, score, threshold=0.5):
    if not isinstance(label, np.ndarray):
        label = np.array(label)
    if not isinstance(score, np.ndarray):
        score = np.array(score)
    
    predictions = (score >= threshold).astype(int)
    correct_predictions = np.sum(predictions == label)
    accuracy = correct_predictions / len(label)
    
    return accuracy