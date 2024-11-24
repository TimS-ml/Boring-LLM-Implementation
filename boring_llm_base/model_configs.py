from typing import Any, Dict, List


# TODO: Update model params to match the boring llm's config settings
MODEL_CONFIGS: List[Dict[str, Any]] = []

llama_3 = [
    # https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json
    dict(
        name="Llama-3.2-1B{}",
        hf_config=dict(org="meta-llama", name="Llama-3.2-1B{}"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=16,
        n_embd=2048,
        n_head=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=8192,
        rope_base=500000,
        rope_adjustments=dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
    # https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json
    dict(
        name="Llama-3.2-3B{}",
        hf_config=dict(org="meta-llama", name="Llama-3.2-3B{}"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=28,
        n_embd=3072,
        n_head=24,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=8192,
        rope_base=500000,
        rope_adjustments=dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
]
MODEL_CONFIGS.extend(llama_3)

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