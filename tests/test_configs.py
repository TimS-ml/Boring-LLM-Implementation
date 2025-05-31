"""Test model configurations compatibility with simplified FFN and PE design"""

import pytest
from boring_llm.base.base_config import BaseConfig
from boring_llm.nn.ffn.main import create_ffn
from boring_llm.nn.pe.main import create_pe


class TestModelConfigs:
    """Test that model configurations work with simplified FFN and PE"""
    
    def test_gpt2_config_loading(self):
        """Test GPT-2 configuration loading"""
        base_config = BaseConfig.from_name("gpt2")
        assert base_config.dim_model == 768
        assert base_config.max_seq_len == 1024
        assert base_config.num_tokens == 50257
        
        # Test FFN config loading
        from boring_llm.base.model_configs import name_to_config
        gpt2_config = name_to_config["gpt2"]
        ffn_config = gpt2_config["ffn_config"]
        
        assert ffn_config["type"] == "standard"
        assert ffn_config["mult_dim"] == 4
        assert ffn_config["activation"] == "gelu"
        
        # Test PE config loading
        pe_config = gpt2_config["pe_config"]
        assert pe_config["type"] == "absolute"
        assert pe_config["max_seq_len"] == 1024
    
    def test_llama_config_loading(self):
        """Test LLaMA configuration loading"""
        base_config = BaseConfig.from_name("MicroLlama-38M")
        assert base_config.dim_model == 1024
        assert base_config.max_seq_len == 2048
        
        from boring_llm.base.model_configs import name_to_config
        llama_config = name_to_config["MicroLlama-38M"]
        
        # Test FFN config
        ffn_config = llama_config["ffn_config"]
        assert ffn_config["type"] == "glu"
        assert ffn_config["mult_dim"] == 2.75
        assert ffn_config["activation"] == "silu"
        
        # Test PE config
        pe_config = llama_config["pe_config"]
        assert pe_config["type"] == "rotary"
        assert pe_config["rotary_percentage"] == 1.0
    
    def test_pythia_config_loading(self):
        """Test Pythia configuration loading"""
        base_config = BaseConfig.from_name("pythia-14m")
        assert base_config.dim_model == 128
        assert base_config.max_seq_len == 512
        
        from boring_llm.base.model_configs import name_to_config
        pythia_config = name_to_config["pythia-14m"]
        
        # Test FFN config
        ffn_config = pythia_config["ffn_config"]
        assert ffn_config["type"] == "standard"
        assert ffn_config["mult_dim"] == 4
        assert ffn_config["activation"] == "gelu"
        
        # Test PE config
        pe_config = pythia_config["pe_config"]
        assert pe_config["type"] == "absolute"
        assert pe_config["max_seq_len"] == 512
    
    def test_config_with_ffn_creation(self):
        """Test creating FFN from config"""
        from boring_llm.base.model_configs import name_to_config
        
        # Test GPT-2 FFN creation
        gpt2_config = name_to_config["gpt2"]
        base_config = gpt2_config["base_config"]
        ffn_config = gpt2_config["ffn_config"]
        
        ffn = create_ffn(
            dim_model=base_config["dim_model"],
            **ffn_config
        )
        
        # Verify FFN parameters through config
        assert ffn.config.dim_model == 768
        assert ffn.config.inner_dim == 768 * 4  # mult_dim = 4
        
        # Test LLaMA FFN creation
        llama_config = name_to_config["MicroLlama-38M"]
        base_config = llama_config["base_config"]
        ffn_config = llama_config["ffn_config"]
        
        ffn = create_ffn(
            dim_model=base_config["dim_model"],
            **ffn_config
        )
        
        # Verify GLU FFN parameters through config
        assert ffn.config.dim_model == 1024
        expected_intermediate = int(1024 * 2.75)
        assert ffn.config.inner_dim == expected_intermediate
    
    def test_config_with_pe_creation(self):
        """Test creating PE from config"""
        from boring_llm.base.model_configs import name_to_config
        
        # Test GPT-2 PE creation
        gpt2_config = name_to_config["gpt2"]
        base_config = gpt2_config["base_config"]
        pe_config = gpt2_config["pe_config"]
        
        pe = create_pe(
            dim_model=base_config["dim_model"],
            **pe_config
        )
        
        # Verify absolute PE through config
        assert pe.config.dim_model == 768
        assert pe.config.max_seq_len == 1024
        
        # Test LLaMA PE creation
        llama_config = name_to_config["MicroLlama-38M"]
        base_config = llama_config["base_config"]
        pe_config = llama_config["pe_config"]
        
        pe = create_pe(
            dim_model=base_config["dim_model"],
            **pe_config
        )
        
        # Verify rotary PE through config
        assert pe.config.dim_model == 1024
        assert pe.config.rotary_percentage == 1.0
    
    def test_all_model_configs_valid(self):
        """Test that all model configurations are valid"""
        from boring_llm.base.model_configs import MODEL_CONFIGS
        
        for config in MODEL_CONFIGS:
            # Test base config loading
            base_config = BaseConfig.from_name(config["name"])
            assert hasattr(base_config, 'dim_model')
            assert hasattr(base_config, 'max_seq_len')
            
            # Test that FFN and PE configs exist
            assert "ffn_config" in config
            assert "pe_config" in config
            
            ffn_config = config["ffn_config"]
            pe_config = config["pe_config"]
            
            # Verify FFN config structure
            assert "type" in ffn_config
            assert ffn_config["type"] in ["standard", "glu"]
            assert "mult_dim" in ffn_config
            assert "activation" in ffn_config
            
            # Verify PE config structure
            assert "type" in pe_config
            assert pe_config["type"] in ["absolute", "rotary", "fixed", "alibi", "none"]


if __name__ == "__main__":
    pytest.main([__file__]) 