---
library_name: peft
license: other
base_model: /root/autodl-tmp/models/Qwen3-VL-8B-Thinking
tags:
- base_model:adapter:/root/autodl-tmp/models/Qwen3-VL-8B-Thinking
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen3vl-8b-aesthetic-lora1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3vl-8b-aesthetic-lora1

This model is a fine-tuned version of [/root/autodl-tmp/models/Qwen3-VL-8B-Thinking](https://huggingface.co//root/autodl-tmp/models/Qwen3-VL-8B-Thinking) on the apdd_aesthetic_thinking dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3520

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.6009        | 0.5428 | 80   | 1.5912          |
| 1.3941        | 1.0814 | 160  | 1.4318          |
| 1.3419        | 1.6243 | 240  | 1.3801          |
| 1.3255        | 2.1628 | 320  | 1.3588          |
| 1.3195        | 2.7057 | 400  | 1.3520          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.2.0
- Pytorch 2.8.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2