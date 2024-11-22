"""
The code is adapted from https://github.com/huggingface/trl.
"""

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
        
    dataset_path: Optional[str] = field(default="", metadata={"help": "dataset path"})
    dev_prompt_path: Optional[str] = field(default="", metadata={"help": "dev prompt path"})

    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    reward_model_batch_size: Optional[int] = field(default=8, metadata={"help": "batch size for RM"})
        
    cache_dir: Optional[str] = field(default="", metadata={"help": "cache_directory"})
    log_with: Optional[str] = field(default="tensorboard", metadata={"help": "use 'wandb' to log with wandb"})
    logging_dir: Optional[str] = field(default="", metadata={"help": "logging directory"})

    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    overall_steps: Optional[int] = field(default=10000, metadata={"help": "number of overall steps"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )

    batched_gen: Optional[bool] = field(default=True, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="outputs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})

    length_incentive: Optional[bool] = field(default=False, metadata={"help": "Use length_incentive as a reward"})
    repetition_penalty: Optional[bool] = field(default=False, metadata={"help": "Use repetition_penalty as a reward"})
    relevance_scaling: Optional[bool] = field(default=False, metadata={"help": "Use relevance_scaling as a reward"})
    reward_branching: Optional[bool] = field(default=False, metadata={"help": "On reward_brancing, relavance scoring by qtype"})
    use_optimizer_setup: Optional[bool] = field(default=False, metadata={"help": "AdamW + Scheduling"})
    warmup_ratio: Optional[float] = field(
        default=0.1,
        metadata={"help": "warmup ratio for linear scheduler with warmup"},
    )
        
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

    exp_name: Optional[str] = field(default="exp", metadata={"help": "Naming for the experiment"})
    rollout_postprocessing: Optional[bool] = field(default=True, metadata={"help": "Do or not rollout postprocessing"})
    chat_template_name: Optional[str] = field(default="alpaca", metadata={"help": "The name of chat template in conversation.py"})
