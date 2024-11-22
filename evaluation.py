import os
import re
import json
import time
import random
import argparse
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
from rouge_score import rouge_scorer

from conversation import get_conv_template

default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def rouge(prediction, ground_truth, xlingual=False):
    scores = default_rouge_scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


class SimpleBatchLoader:

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.n_iter = int(len(data) / batch_size)
        if len(data) % batch_size != 0:
            self.n_iter += 1

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for idx in range(self.n_iter):
            indices = list(range(idx * self.batch_size,
                                 (idx + 1) * self.batch_size))
            batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
            yield indices, batch


def construct_prompt(template, msg):
    template.messages = []
    template.append_message(template.roles[0], msg)
    template.append_message(template.roles[-1], None)
    return template.get_prompt()

    
def run_alpaca_eval(model,
                    tokenizer,
                    TEMPLATE,
                    eval_set,
                    model_name,
                    output_path,
                    batch_size=8,
                    top_p=1.0,
                    temperature=0.7,
                    max_new_tokens=512):

    model_name = model_name.split("/")[-1]
    if os.path.exists(f"{output_path}/alpaca-eval-{model_name}.json"):
        idx = 1
        _output_path = f"{output_path}/alpaca-eval-{model_name}-{idx}.json"
        while os.path.exists(_output_path):
            idx += 1
            _output_path = f"{output_path}/alpaca-eval-{model_name}-{idx}.json"
        output_path = _output_path
        model_name = f"{model_name}-{idx}"
    else:
        output_path = f"{output_path}/alpaca-eval-{model_name}.json"

    data_loader = SimpleBatchLoader(eval_set, batch_size=batch_size)
    for idx, (indices, batch) in enumerate(data_loader):
        prompts = [construct_prompt(TEMPLATE, instance['instruction']) for instance in batch]
        inputs = tokenizer(prompts,
                           truncation=True,
                           padding="longest",
                           return_tensors="pt")
        outputs = model.generate(input_ids=inputs.input_ids.to(model.device),
                                 attention_mask=inputs.attention_mask.to(model.device),
                                 do_sample=True,
                                 use_cache=True,
                                 top_p=top_p,
                                 temperature=temperature,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id,
                                 max_new_tokens=max_new_tokens)
        for i, prompt, output in zip(indices, prompts, tokenizer.batch_decode(outputs, skip_special_tokens=True)):
            prompt = prompt.replace("<s>", "").replace("</s>", " ")
            text_output = output.replace(prompt, "")
            if TEMPLATE.stop_str and text_output.find(TEMPLATE.stop_str) > 0:
                text_output = text_output[: text_output.find(TEMPLATE.stop_str)]

            text_output = text_output.strip(" \n")
            eval_set[i]['output'] = text_output
            eval_set[i]['generator'] = model_name
        print(f"AlpacaEval: {idx}/{len(data_loader)}")
    json.dump(eval_set, open(output_path, "w", encoding="utf-8"))


def run_superni(model,
                tokenizer,
                TEMPLATE,
                target_dir,
                device=0,
                batch_size=8):
    
    validations = open(f"{target_dir}/splits/default/test_tasks.txt").read().splitlines()
    all_prompts = []
    all_outputs = []
    for file in os.listdir(f"{target_dir}/tasks"):
        if file.replace(".json", "") not in validations:
            continue

        data = json.load(open(f"{target_dir}/tasks/{file}", "r", encoding="utf-8"))
        for idx, instance in enumerate(data['Instances'][:100]):
            p = data['Definition'][0] + f"\n\n{instance['input']}"
            prompt = construct_prompt(TEMPLATE, p)
            all_prompts.append(prompt)
            all_outputs.append(instance['output'][0])
    tokenizer.padding_side = "left"
    all_input_dicts = tokenizer(
                all_prompts,
                return_tensors="pt",
                padding="longest",
                max_length=512,
                truncation=True,
            )
    dataset = TensorDataset(all_input_dicts['input_ids'], all_input_dicts['attention_mask'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_predictions = []
    for idx, batch in enumerate(loader):
        input_ids, attention_mask = batch
        outputs = model.generate(input_ids=input_ids.to(model.device),
                                 attention_mask=attention_mask.to(model.device),
                                 do_sample=False,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.pad_token_id,
                                 max_new_tokens=128)

        for output, input_ in zip(tokenizer.batch_decode(outputs, skip_special_tokens=True),  tokenizer.batch_decode(input_ids, skip_special_tokens=True)):
            all_predictions.append(output.replace(input_, ""))
        print(f"SuperNI {idx}/{len(loader)}", end="\r")

    rouges = []
    for pred, output in zip(all_predictions, all_outputs):
        rouges.append(rouge(pred, output))
        
    rouge_score = np.mean(rouges)
    print(rouge_score)
    result = {'rouge-L': rouge_score, 'predictions': all_predictions}
    return result
    

def load_model(base_model_name, peft_model_name, cache_dir=None, device=0):

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    )
    
    if peft_model_name:
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model.eval()
        model = model.merge_and_unload()
        try:
            tokenizer = AutoTokenizer.from_pretrained(peft_model_name)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(peft_model_name)
    else:
        model = base_model
        model.eval()
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
    
    if "vicuna" in base_model_name:
        TEMPLATE = get_conv_template("vicuna_v1.1")
    elif "alpaca" in base_model_name:
        TEMPLATE = get_conv_template("alpaca")
    elif "Llama-2" in base_model_name:
        TEMPLATE = get_conv_template("llama-2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif "llama-7b" in base_model_name:
        TEMPLATE = get_conv_template("llama-2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif "tulu" in base_model_name:
        TEMPLATE = get_conv_template("tulu")
    elif "mistral" in base_model_name:
        TEMPLATE = get_conv_template("zephyr")
    elif "zephyr" in base_model_name:
        TEMPLATE = get_conv_template("zephyr")
    else:
        TEMPLATE = get_conv_template("almost")
    
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        model.config.pad_token_id = tokenizer.unk_token_id
    
    return model, tokenizer, TEMPLATE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default=None)
    parser.add_argument("--peft_model_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--baseline_model_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--evaluations", nargs="+", default=[])
    args = parser.parse_args()
    
    model, tokenizer, TEMPLATE = load_model(args.base_model_name, args.peft_model_name, args.cache_dir, args.device)
    print("Model is loaded!")
    save_dir = args.peft_model_name if args.peft_model_name else args.base_model_name
    if not os.path.exists(save_dir):
        save_dir = f"outputs/{args.base_model_name.replace('/', '_')}"
        os.makedirs(save_dir, exist_ok=True)

    ## Alpaca Eval
    if "alpaca" in args.evaluations or not args.evaluations:
        alpaca_eval_set = json.load(open("data/alpaca_eval.json"))
        model_name = args.peft_model_name if args.peft_model_name else args.base_model_name.split("/")[-1]
        run_alpaca_eval(model, tokenizer, TEMPLATE, alpaca_eval_set, model_name, save_dir, args.batch_size)
    
    ## SuperNI Eval
    if "superni" in args.evaluations or not args.evaluations:
        result = run_superni(model, tokenizer, TEMPLATE, "data/natural-instructions", args.device, batch_size=args.batch_size)
        json.dump(result, open(os.path.join(save_dir, "superni_eval.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

