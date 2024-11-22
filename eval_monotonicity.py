'''
Rethinking-Proxy-Reward
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0
'''

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

from conversation import get_conv_template
from reward_model import ReversedEngineeredRewardForInference, StarlingRMForInference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_prefix", type=str, default=None)
    
    data = json.load(open("data/alpaca_eval.json"))
    template = get_conv_template("alpaca")
    
    proxy_rm = ReversedEngineeredRewardForInference(
        template,
        length_incentive=True,
        repetition_penalty=True,
        relevance_scaling=True,
        reward_branching=True,
        do_strip=True)
    golden_rm = StarlingRMForInference(template, device="cuda")

    result_path = [f"{args.output_dir}/{args.model_prefix}_step_{step}/alpaca-eval-{args.model_prefix}_step_{step}.json" \
                   for step in range(500, 5500, 500)]
    
    proxy_rewards = []
    golden_rewards = []

    for rid, path in enumerate(result_path):
        result = json.load(open(path))
        proxy_reward = []
        golden_reward = []
        for idx, (d, p) in enumerate(zip(data, result)):

            score = proxy_rm.get_reward(
                {
                    "query": [d['instruction']],
                    "response": [p['output']],
                    "qtype": [d['qtype']],
                    "reference": [d['output']]
                }
            )
            proxy_reward.append(score[0])

            score = golden_rm.get_reward(
                {
                    "query": [d['instruction']],
                    "response": [p['output']],
                    "qtype": [d['qtype']],
                    "reference": [d['output']]
                }
            )
            golden_reward.append(score)
            
            if idx % 10 == 0:
                print(f"[{rid}] {idx}/{len(result)}")

        proxy_rewards.append(np.mean(proxy_reward))
        golden_rewards.append(np.mean(golden_reward))
        
    spearman = spearmanr(proxy_rewards, golden_rewards)
    print(spearman)
