from boring_llm.nn.pe.core import *


class PositionalEmbeddingFactory:
    """Factory for creating positional embedding modules"""
    
    @staticmethod
    def create(embedding_type: str, **kwargs):
        """
        Create a positional embedding module
        
        Args:
            embedding_type: Type of positional embedding ('fixed', 'absolute')
            **kwargs: Arguments for the specific embedding type
        
        Returns:
            A positional embedding module
        """
        if embedding_type == 'fixed':
            return FixedPositionalEmbedding(**kwargs)
        elif embedding_type == 'absolute':
            return AbsolutePositionalEmbedding(**kwargs)
        else:
            raise ValueError(f"Unknown positional embedding type: {embedding_type}")