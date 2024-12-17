# Copyright (c) 2024 Harim Kang
# SPDX-License-Identifier: MIT

"""
Text embedding model implementation using Qwen.
"""

from typing import List, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling


class EmbeddingModel:
    """Text embedding model using Qwen."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B",
        device: str = "cuda",
        max_length: int = 512,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: Name or path of the Qwen model
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum sequence length for tokenization
            normalize_embeddings: Whether to L2-normalize the embeddings
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # Load tokenizer and model with trust_remote_code=True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Configure model loading based on device
        if self.device == "cuda":
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use half precision
                device_map="balanced",  # Balanced allocation across GPUs
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            ).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size

    def _mean_pooling(self, model_output: BaseModelOutputWithPooling, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings.

        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask from tokenizer

        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: Union[str, List[str]], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for the given texts.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Number of texts to process at once

        Returns:
            numpy array of embeddings with shape (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize and generate embeddings
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean of last hidden states as embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)

                # Normalize if requested
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Move to CPU and convert to numpy array
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)

            # Clear GPU cache
            torch.cuda.empty_cache()

        # Combine all batches
        return np.vstack(all_embeddings)

    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the embeddings.

        Returns:
            Embedding dimension
        """
        return self.embedding_dim
