"""
We utilize code snippet from https://huggingface.co/Nexusflow/Starling-RM-34B.
We also refer to https://huggingface.co/facebook/contriever and https://huggingface.co/01-ai/Yi-34B-Chat.
"""

import os
import re
import math
import json
import nltk
import torch
from torch import nn
from transformers import LlamaPreTrainedModel, LlamaModel
from transformers import AutoTokenizer, AutoModel


class ContrieverModelForInference:
    
    def __init__(self, model_name_or_path='facebook/contriever', batch_size=4, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.truncation_side = "left"
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.batch_size = batch_size
        self.model.eval()
        self.model.to(device)
        
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def to(self, device):
        self.model.to(device)
        
    def extract_raw_query(self, batch):
        query = []
        response = []
        for q in batch["query"]:
            q = q.replace(self.template.system, "")
            q = q.replace(self.template.roles[0], " ")
            q = q.replace("### Input:\n", " ").replace("\n ", "\n")
            q = q.replace(self.template.roles[-1], " ").strip("\n: ")
            query.append(q)
        return dict(query=query, response=batch["response"])

    def get_reward(self, data):
        if len(data["query"]) <= self.batch_size:
            return self.batch_inference(data)

        all_scores = []
        for idx in range(0, len(data["query"]), self.batch_size):
            batch = {
                "query": data["query"][idx:idx + self.batch_size],
                "response": data["response"][idx:idx + self.batch_size],
            }
            scores = self.batch_inference(batch)
            all_scores.extend(scores)
        return all_scores

    def batch_inference(self, batch):
        batched = batch["query"] + batch["response"]
        x = self.tokenizer(
                batched,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                padding="longest",
                return_tensors="pt"
        )
        
        for ins in x:
            x[ins] = x[ins].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**x)
        bs = outputs[0].shape[0] // 2
        embeddings = self.mean_pooling(outputs[0], x["attention_mask"])
        
        query_embeddings = embeddings[:bs]
        response_embeddings = embeddings[bs:]
        batch_scores = query_embeddings @ response_embeddings.T
        scores = batch_scores.cpu().masked_select(torch.eye(bs).bool())
        return scores.tolist()

    
class ReversedEngineeredRewardForInference:
    
    def __init__(self,
                 template,
                 max_sequence_length=512,
                 length_incentive=True,
                 repetition_penalty=False,
                 relevance_scaling=False,
                 reward_branching=False,
                 do_strip=True,
                 device=0):
        self.template = template
        self.max_sequence_length = max_sequence_length
        self.length_incentive = length_incentive
        self.repetition_penalty = repetition_penalty
        self.relevance_scaling = relevance_scaling
        if relevance_scaling:
            self.relevance_model = ContrieverModelForInference(device=f"cuda:{device}")
        self.reward_branching = reward_branching
        self.do_strip = do_strip
        role_prefix = [role + ":" if "colon" in template.sep_style.name.lower() else role for role in template.roles]
        self.strip_strings = role_prefix + ["\n\n\n\n", "\n\n--", "\n\n____"]
        
    def linear_interpolation(self,
                             value: float,
                             input_start: float = 0.,
                             input_end: float = 3.5,
                             output_start: float = 0.,
                             output_end: float = 5.,
                            ):
        i_range = input_end - input_start
        o_range = output_end - output_start
        return output_start + ((value - input_start) / i_range) * o_range
        
    def extract_raw_query(self, batch):
        query = []
        response = []
        for q, qtype, refer in zip(batch["query"], batch["qtype"], batch["reference"]):
            if self.reward_branching and qtype == "CONSTRAINED":
                q = refer
            else:
                q = q.replace(self.template.system, "")
                qs = q.split(self.template.roles[0])
                q = ""
                for idx, qq in enumerate(qs):
                    if not qq.strip():
                        continue

                    qq = qq.split(self.template.roles[-1])[0]
                    qq = qq.replace("### Input:\n", " ").replace("</s>", "").replace("\n ", "\n")
                    qq = qq.replace(" : ", " ").strip("\n: ")
                    q += qq + " "
                q = re.sub(r'\s+', ' ', q).strip()
            query.append(q)
        return dict(query=query, response=batch["response"], qtype=batch["qtype"])

    def _combine_scores(self, batch, length_scores, trigram_scores):
        batch = self.extract_raw_query(batch)
        if self.relevance_scaling:
            relevance_scores = self.relevance_model.get_reward(batch)
        else:
            relevance_scores = [1. for _ in range(len(length_scores))]

        final_scores = []
        for rs, ls, ts, qtype in zip(relevance_scores, length_scores, trigram_scores, batch["qtype"]):
            if self.relevance_scaling and self.reward_branching and qtype == "CONSTRAINED":
                score = self.linear_interpolation(rs, output_end=self.max_sequence_length / 100)
            else:
                score = ls * rs

            if self.repetition_penalty:
                final_scores.append(score * ts)
            else:
                final_scores.append(score)
                
        return final_scores
    
    def get_length_incentive(self, text):
        if self.do_strip and self.strip_strings:
            STOP_SEQUENCE = '[STOP_SEQUENCE]'
            text = re.sub(f"({'|'.join(self.strip_strings)})", STOP_SEQUENCE, text.strip("\n "))
            text = text.split(STOP_SEQUENCE, 1)[0].strip("\n ")
        tokens = nltk.wordpunct_tokenize(text)
        length_incentive = len(tokens) / 100

        trigrams = []
        for trigram in nltk.ngrams([str(token) for token in tokens], 3):
            trigrams.append(" ".join(trigram))
        if trigrams:
            repetition_penalty = len(set(trigrams)) / len(trigrams)
        else:
            repetition_penalty = 1.0
        
        return length_incentive, repetition_penalty
        
    def get_reward(self, batch):
        length_scores = []
        trigram_scores = []
        for sample in batch["response"]:
            length_incentive, repetition_penalty = self.get_length_incentive(sample)

            if not self.length_incentive:
                length_incentive = 1.

            length_scores.append(length_incentive)
            trigram_scores.append(repetition_penalty)
        scores = self._combine_scores(batch, length_scores, trigram_scores)
        return scores
    
    
### Starling-RM
### https://huggingface.co/Nexusflow/Starling-RM-34B

class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.PAD_ID = 0
        # Initialize weights and apply final processing
        self.post_init()

    def get_device(self):
        return self.transformer.device

    def forward(
          self,
          input_ids=None,
          past_key_values=None,
          attention_mask=None,
          position_ids=None,
      ):
        transformer_outputs = self.transformer(
          input_ids,
          attention_mask=attention_mask,
          position_ids=position_ids,
          output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        bs = int(input_ids.shape[0])
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        scores = torch.stack(scores)
        return scores
    
    
class StarlingRMForInference():
    
    def __init__(self, template, batch_size=4, model_max_length=2048, device="cuda"):
        self.model = LlamaForSequenceClassification.from_pretrained("Nexusflow/Starling-RM-34B",
                                                                    device_map='auto',
                                                                    torch_dtype=torch.bfloat16)
        self.template = template
        self.model.eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B-Chat",
                                                        model_max_length=model_max_length,
                                                        padding_side="right",
                                                        use_fast=False)
        self.tokenizer.truncation_side = "left"
        self.batch_size = batch_size

    def to(self, device):
        self.model.to(device)
        
    def extract_raw_query(self, batch):
        query = []
        response = []
        for q in batch["query"]:
            q = q.replace(self.template.system, "")
            q = q.replace(self.template.roles[0], " ")
            q = q.replace("### Input:\n", " ").replace("\n ", "\n")
            q = q.replace(self.template.roles[-1], " ").strip("\n: ")
            query.append(f"\n\nHuman: {q}\n\nAssistant: ")
        return dict(query=query, response=batch["response"])

    def get_reward(self, data):
        data = self.extract_raw_query(data)
        if len(data["query"]) <= self.batch_size:
            return self.batch_inference(data)

        all_scores = []
        for idx in range(0, len(data["query"]), self.batch_size):
            batch = {
                "query": data["query"][idx:idx + self.batch_size],
                "response": data["response"][idx:idx + self.batch_size],
            }
            scores = self.batch_inference(batch)
            all_scores.extend(scores)
        return all_scores

    def batch_inference(self, batch):
        PROMPT = "<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"
        batched = []
        for q, a in zip(batch['query'], batch['response']):
            batched.append(PROMPT.format(query=q, assistant=a))
        x = self.tokenizer(
                batched,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                padding="longest",
                add_special_tokens=False,
                return_tensors="pt"
        )
        
        for ins in x:
            x[ins] = x[ins].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**x)
        return outputs.squeeze().cpu().tolist()