"""
The code is adapted from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py.
"""

import re
import os
import json
import torch
import nltk
import math
import random
from tqdm import tqdm
import numpy as np
import pandas as pd

tqdm.pandas()

from torch.optim import Adam
from transformers import LlamaTokenizer, HfArgumentParser, get_constant_schedule_with_warmup
from datasets import load_dataset, Dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
from peft import LoraConfig
from accelerate import Accelerator

from rich.console import Console
from rich.table import Table
from dataclasses import asdict

from reward_model import ReversedEngineeredRewardForInference, StarlingRMForInference
from conversation import get_conv_template

from training_args import ScriptArguments
from utils import strip_response_tensors, significant


def build_dataset_from_file(
    tokenizer,
    dataset_path,
    chat_template,
    max_sequence_length=512
):
    data = json.load(open(dataset_path, "r", encoding="utf-8"))
    train_dataset = Dataset.from_list(data)
    original_columns = train_dataset.column_names
    num_proc = 8
    
    def format_chat_template(msg, chat_template):
        chat_template.messages = []
        chat_template.append_message(chat_template.roles[0], msg)
        chat_template.append_message(chat_template.roles[-1], None)
        return chat_template.get_prompt()

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
            "reference": [],
            "qtype": []
        }
        for query in examples["input"]:
            query = format_chat_template(query, chat_template)
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])
        
        if examples.get("qtype"):
            for answer, qtype in zip(examples["output"], examples["qtype"]):
                new_examples["reference"].append(answer)
                new_examples["qtype"].append(qtype)
                
        if not new_examples.get("qtype"):
            new_examples.pop("qtype")
            
        if not new_examples.get("reference"):
            new_examples.pop("reference")

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < max_sequence_length, batched=False)

    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def evaluate(ppo_trainer, dev_dataloader, tokenizer, output_length_sampler, reward_model, generation_kwargs):
    overall_rewards = []
    overall_eval_step = len(dev_dataloader)
    for epoch, batch in tqdm(enumerate(dev_dataloader)):
    
        question_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            batch_size=len(question_tensors),  # default: 4
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        rewards = [score for score in reward_model.get_reward(batch)]

        columns = ["prompt", "response", "reward"]
        rich_table = Table(*columns, title=f"[{epoch}/{overall_eval_step}] Eval Avg reward: {np.mean(rewards).item()}", show_lines=True)
        for ix in range(min(3, len(question_tensors))):
            row = [batch["query"][ix], batch["response"][ix], str(significant(rewards[ix]))]
            rich_table.add_row(*row)

        if rich_table.row_count:
            try:
                Console().print(rich_table)
            except:
                pass
        overall_rewards.extend(rewards)
    
    logs = {}
    logs["eval/reward_mean"] = np.mean(overall_rewards).item()
    logs["eval/reward_std"] = np.std(overall_rewards).item()
    ppo_trainer.accelerator.log(
        logs,
        step=ppo_trainer.current_step if ppo_trainer.config.log_with == "tensorboard" else None
    )
    return logs


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
        
    current_device = Accelerator().local_process_index
    if current_device == 0 and not os.path.exists(script_args.output_dir):
        os.mkdir(script_args.output_dir)
        json.dump(asdict(script_args), open(f"{script_args.output_dir}/training_config.json", "w", encoding="utf-8"), indent=2)
        
    if current_device == 0 and not os.path.exists(script_args.logging_dir):
        os.mkdir(script_args.logging_dir)
        
    config = PPOConfig(
        steps=script_args.overall_steps,
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1.0,
        log_with="tensorboard",
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        early_stopping=False,
        target_kl=0.1,
        seed=script_args.seed,
        init_kl_coef=script_args.init_kl_coef,
        adap_kl_ctrl=True,
        use_score_scaling=True,
        use_score_norm=True,
        score_clip=10,
        remove_unused_columns=False
    )
    set_seed(config.seed)
    config.project_kwargs['project_dir'] = script_args.logging_dir
    config.project_kwargs['logging_dir'] = script_args.logging_dir
    tokenizer = LlamaTokenizer.from_pretrained(config.model_name, truncation="left", cache_dir=script_args.cache_dir)
    
    chat_template = get_conv_template(script_args.chat_template_name)
    dataset = build_dataset_from_file(tokenizer, script_args.dataset_path, chat_template, script_args.output_max_length)
    dev_dataset = build_dataset_from_file(tokenizer, script_args.dev_prompt_path, chat_template, script_args.output_max_length)
    
    lora_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'v_proj']
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        device_map={"": current_device},
        cache_dir=script_args.cache_dir,
        peft_config=lora_config,
    )

    reward_model = ReversedEngineeredRewardForInference(
        chat_template,
        max_sequence_length=script_args.output_max_length,
        length_incentive=script_args.length_incentive,
        repetition_penalty=script_args.repetition_penalty,
        relevance_scaling=script_args.relevance_scaling,
        reward_branching=script_args.reward_branching,
        do_strip=script_args.rollout_postprocessing,
        device=current_device
    )
    
    if script_args.use_optimizer_setup:
        optimizer =  Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=script_args.learning_rate,
        )
        lr_scheduler = get_constant_schedule_with_warmup(optimizer,
                                                         script_args.overall_steps * script_args.warmup_ratio)

    else:
        optimizer = None
        lr_scheduler = None

    ref_model = None
    num_shared_layers = None
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
    dev_dataloader = ppo_trainer.prepare_dataloader(dev_dataset, collator)
    
    role_prefix = [role + ":" if "colon" in chat_template.sep_style.name.lower() else role for role in chat_template.roles]
    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "max_new_tokens": script_args.output_max_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    output_length_sampler = None
    highest_eval_reward = -100000
    while ppo_trainer.current_step < script_args.overall_steps:

        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            if ppo_trainer.current_step >= script_args.overall_steps:
                break

            question_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                batch_size=len(question_tensors),  # default: 4
                **generation_kwargs,
            )
           
            if script_args.rollout_postprocessing:
                response_tensors, batch["response"] = strip_response_tensors(
                    response_tensors,
                    tokenizer,
                    role_prefix
                )
            else:
                batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                
            rewards = [torch.tensor(score - script_args.reward_baseline) for score in reward_model.get_reward(batch)]

        #     # Run PPO step
            if script_args.gradient_checkpointing:
                ppo_trainer.model.train()
                if hasattr(ppo_trainer.model, 'module'):
                    ppo_trainer.model.module.gradient_checkpointing_enable()
                    ppo_trainer.model.module.pretrained_model.enable_input_require_grads()
                else:
                    ppo_trainer.model.gradient_checkpointing_enable()
                    ppo_trainer.model.pretrained_model.enable_input_require_grads()
            try:
                stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
            except Exception as e:
                print(str(e))
                print("Something wrong in trainer step! Skip the batch...")
                continue

            ppo_trainer.log_stats(stats, batch, rewards)

            columns = ["prompt", "response", "reward"]
            rich_table = Table(*columns, title=f"{script_args.exp_name} Avg reward @ {ppo_trainer.current_step}: {np.mean(rewards).item()}", show_lines=True)
            for ix in range(min(3, len(question_tensors))):
                row = [batch["query"][ix], batch["response"][ix], str(significant(rewards[ix]))]
                rich_table.add_row(*row)
                
            if current_device == 0 and rich_table.row_count:
                try:
                    Console().print(rich_table)
                except:
                    print("skip logging..")
                    pass
                
            if script_args.gradient_checkpointing:
                if hasattr(ppo_trainer.model, 'module'):
                    ppo_trainer.model.module.gradient_checkpointing_disable()
                    ppo_trainer.model.module.pretrained_model.disable_input_require_grads()
                else:
                    ppo_trainer.model.gradient_checkpointing_disable()
                    ppo_trainer.model.pretrained_model.disable_input_require_grads()

            if script_args.save_freq and ppo_trainer.current_step and ppo_trainer.current_step % script_args.save_freq == 0:
                
                eval_result = evaluate(ppo_trainer, dev_dataloader, tokenizer, output_length_sampler, reward_model, generation_kwargs)
                
                if eval_result["eval/reward_mean"] >= highest_eval_reward:
                    highest_eval_reward = eval_result["eval/reward_mean"]

                ppo_trainer.save_pretrained(script_args.output_dir + f"_step_{ppo_trainer.current_step}")
        ppo_trainer.save_pretrained(script_args.output_dir)
        with open(f"{script_args.output_dir}/step_info.txt", "a", encoding="utf-8") as f:
            f.write(f"{ppo_trainer.current_step}\n")
    ppo_trainer.save_pretrained(script_args.output_dir + "_final")