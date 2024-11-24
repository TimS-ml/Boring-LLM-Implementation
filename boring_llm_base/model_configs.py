from typing import Any, Dict, List
from copy import deepcopy


# TODO: Update model params to match the boring llm's config settings
# TODO: Test nested config, like pythia
MODEL_CONFIGS: List[Dict[str, Any]] = []


gpt = [
    # https://huggingface.co/openai-community/gpt2/blob/main/config.json
    dict(
        name="gpt2",
        hf_config=dict(org="openai-community", name="gpt2"),
        base_config=dict(
            d_model=768,
            num_tokens=50257,
            dropout=0.1
        ),
        attention_config=dict(
            dim_head=64,
            num_heads=12,
            bias=True
        ),
        ffn_config=dict(
            mlp_class_name="GptNeoxMLP",
            intermediate_size=3072,
            bias=True,
            activation_function="gelu_new"
        ),
        block_size=1024,
        padded_vocab_size=50257,
        n_layer=12,
        parallel_residual=True,
        norm_class_name="LayerNorm",
        norm_eps=1e-5
    ),
]
MODEL_CONFIGS.extend(gpt)


tiny_llama = [
    # https://huggingface.co/keeeeenw/MicroLlama/blob/main/config.json
    dict(
        name="MicroLlama-38M",
        hf_config=dict(org="keeeeenw", name="MicroLlama-38M"),
        base_config=dict(
            d_model=1024,
            num_tokens=32000,
            dropout=0.0
        ),
        attention_config=dict(
            dim_head=64,
            num_heads=16,
            n_query_groups=4,
            rotary_percentage=1.0,
            bias=False
        ),
        ffn_config=dict(
            mlp_class_name="LLaMAMLP",
            intermediate_size=5632,
            bias=False
        ),
        block_size=2048,
        padded_vocab_size=32000,
        n_layer=12,
        parallel_residual=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-5,
        rope_base=10000
    ),
]
MODEL_CONFIGS.extend(tiny_llama)


llama_3 = [
    # https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json
    dict(
        name="Llama-3.2-1B{}",
        hf_config=dict(org="meta-llama", name="Llama-3.2-1B{}"),
        base_config=dict(
            d_model=2048,
            num_tokens=128000,
            dropout=0.1
        ),
        attention_config=dict(
            dim_head=64,
            num_heads=32,
            n_query_groups=8,
            rotary_percentage=1.0,
            bias=False
        ),
        ffn_config=dict(
            mlp_class_name="LLaMAMLP",
            intermediate_size=8192,
            bias=False
        ),
        block_size=131072,
        padded_vocab_size=128256,
        n_layer=16,
        parallel_residual=False,
        norm_class_name="RMSNorm",
        rope_base=500000,
        rope_adjustments=dict(
            factor=32.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_seq_len=8192
        )
    ),
    # # https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json
    # dict(
    #     name="Llama-3.2-3B{}",
    #     hf_config=dict(org="meta-llama", name="Llama-3.2-3B{}"),
    #     block_size=131072,
    #     vocab_size=128000,
    #     padded_vocab_size=128256,
    #     n_layer=28,
    #     n_embd=3072,
    #     n_head=24,
    #     n_query_groups=8,
    #     rotary_percentage=1.0,
    #     parallel_residual=False,
    #     bias=False,
    #     norm_class_name="RMSNorm",
    #     mlp_class_name="LLaMAMLP",
    #     intermediate_size=8192,
    #     rope_base=500000,
    #     rope_adjustments=dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    # ),
]
# MODEL_CONFIGS.extend(llama_3)
for c in llama_3:
    for kind in ("", "-Instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        MODEL_CONFIGS.append(copy)


phi = [
    # Phi-2
    dict(
        name="phi-2",
        hf_config=dict(org="microsoft", name="phi-2"),
        d_model=2560,
        num_tokens=50257,
        padded_vocab_size=51200,
        block_size=2048,
        n_layer=32,
        rotary_percentage=0.4,
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
    ),
]
MODEL_CONFIGS.extend(phi)


pythia = [
    # https://huggingface.co/EleutherAI/pythia-14m/blob/main/config.json
    dict(
        name="pythia-14m",
        hf_config=dict(org="EleutherAI", name="pythia-14m"),
        block_size=512,
        n_layer=6,
        n_embd=128,
        n_head=4,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-31m/blob/main/config.json
    dict(
        name="pythia-31m",
        hf_config=dict(org="EleutherAI", name="pythia-31m"),
        block_size=1024,
        n_layer=6,
        n_embd=256,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json
    dict(
        name="pythia-70m",
        hf_config=dict(org="EleutherAI", name="pythia-70m"),
        block_size=2048,
        n_layer=6,
        n_embd=512,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-160m/blob/main/config.json
    dict(
        name="pythia-160m",
        hf_config=dict(org="EleutherAI", name="pythia-160m"),
        block_size=2048,
        n_layer=12,
        n_embd=768,
        n_head=12,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-410m/blob/main/config.json
    dict(
        name="pythia-410m",
        hf_config=dict(org="EleutherAI", name="pythia-410m"),
        block_size=2048,
        n_layer=24,
        n_embd=1024,
        n_head=16,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-1b/blob/main/config.json
    dict(
        name="pythia-1b",
        hf_config=dict(org="EleutherAI", name="pythia-1b"),
        block_size=2048,
        n_embd=2048,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-1.4b/blob/main/config.json
    dict(
        name="pythia-1.4b",
        hf_config=dict(org="EleutherAI", name="pythia-1.4b"),
        block_size=2048,
        n_layer=24,
        n_embd=2048,
        n_head=16,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-2.8b/blob/main/config.json
    dict(
        name="pythia-2.8b",
        hf_config=dict(org="EleutherAI", name="pythia-2.8b"),
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        padding_multiple=128,
    ),
]
# MODEL_CONFIGS.extend(pythia)
for c in pythia:
    # "pythia-14m" and "pythia-31m" don't have deduped version
    if c["name"] in ("pythia-14m", "pythia-31m"):
        continue
    copy = deepcopy(c)
    copy["name"] = f"{c['name']}-deduped"
    copy["hf_config"]["name"] = f"{c['hf_config']['name']}-deduped"
    MODEL_CONFIGS.append(copy)