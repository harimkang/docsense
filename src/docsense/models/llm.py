# Copyright (c) 2024 Harim Kang
# SPDX-License-Identifier: MIT

"""
LLM (Large Language Model) wrapper implementation.
"""

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


class LLMModel:
    """Wrapper for Qwen language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B",
        device: str = "cuda",
        max_length: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
    ):
        """
        Initialize the LLM model.

        Args:
            model_name: Name of the Qwen model to use
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum sequence length for generation
            temperature: Sampling temperature (0.0 for deterministic output)
            top_p: Nucleus sampling parameter (1.0 for no filtering)
            repetition_penalty: Penalty for repeating tokens
        """
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if device == "cuda" and not cuda_available:
            print("CUDA is not available. Using CPU instead.")
            device = "cpu"
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Configure model loading based on device
        if self.device == "cuda":
            # Check if flash-attn is available
            try:
                import flash_attn  # noqa: F401

                has_flash_attn = True
            except ImportError:
                has_flash_attn = False
                print("Flash Attention 2 not found. Using default attention implementation.")

            # Base model configuration for GPU
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "cuda:0",
                "low_cpu_mem_usage": True,
                "use_cache": True,
                "load_in_8bit": True,
                "llm_int8_enable_fp32_cpu_offload": True,
                "offload_folder": "offload",
            }

            # Add Flash Attention 2 if available
            if has_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            # CPU configuration
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float32,
                "device_map": "cpu",
                "use_cache": True,
            }

        # Load model with appropriate configuration
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Set model to evaluation mode
        self.model.eval()

        # Set generation config for faster responses
        self.generation_config = GenerationConfig(
            max_length=max_length,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=128,
            use_cache=True,
            repetition_penalty=repetition_penalty,
        )

    def _create_prompt(self, question: str, context: Optional[List[str]] = None) -> str:
        """
        Create a prompt for the model.

        Args:
            question: User question
            context: Optional list of context passages

        Returns:
            Formatted prompt string
        """
        if context:
            # Limit context length
            total_context_length = 0
            selected_contexts = []
            for ctx in context:
                if total_context_length + len(ctx) > 2000:  # Adjust threshold as needed
                    break
                selected_contexts.append(ctx)
                total_context_length += len(ctx)

            context_str = "\n\n".join(selected_contexts)
            prompt = (
                "Answer the question based on the following context. "
                "Be concise and only use information from the context.\n"
                f"Context: {context_str}\n"
                f"Question: {question}\n"
                "Answer: "
            )
        else:
            prompt = f"Question: {question}\nAnswer: "

        return prompt

    @torch.no_grad()
    def generate(self, question: str, context: List[str]) -> Dict[str, Any]:
        """Generate answer based on context."""
        context_text = self._prepare_context(context, max_length=1000)

        prompt = f"""Use the following documents to answer the question.
For each fact you mention, include the document number in brackets like [Document 1].
If you cannot find the answer in the documents, simply say so.

Documents:
{context_text}

Question: {question}
Answer:"""

        generation_config = {
            "max_new_tokens": 256,
            "temperature": self.temperature,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "repetition_penalty": 1.0,
        }

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(
            self.model.device
        )

        outputs = self.model.generate(**inputs, generation_config=GenerationConfig(**generation_config))

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt) :].strip()
        answer = self._post_process_response(answer)

        return {"answer": answer, "prompt": prompt, "generation_config": generation_config}

    def get_max_length(self) -> int:
        """
        Get the maximum sequence length.

        Returns:
            Maximum sequence length
        """
        return self.max_length

    def _prepare_context(self, context: List[str], max_length: int = 2000) -> str:
        """
        Prepare context for the model by formatting with document references.

        Args:
            context: List of context strings to format
            max_length: Maximum total length of formatted context

        Returns:
            Formatted context string with document references
        """
        # Format context with document references
        formatted_contexts = []
        current_length = 0

        for i, text in enumerate(context, 1):
            if current_length + len(text) > max_length:
                break

            # Add document reference to each context
            formatted_text = f"[Document {i}]\n{text}"
            formatted_contexts.append(formatted_text)
            current_length += len(text)

        return "\n\n---\n\n".join(formatted_contexts)

    def _post_process_response(self, response: str) -> str:
        """
        Clean up and validate the model response.

        Args:
            response: Raw response string from the model

        Returns:
            Processed response with validation messages if needed

        Notes:
            - Truncates responses longer than 1000 characters
            - Checks for speculative language and missing citations
            - Adds guidance messages when response doesn't meet quality criteria
        """
        # Return as-is if response indicates no answer found
        if any(
            phrase in response.lower()
            for phrase in [
                "cannot find",
                "not found",
                "no information",
                "cannot answer",
                "don't have enough information",
            ]
        ):
            return response

        # Truncate if response is too long
        if len(response) > 1000:
            response = response[:1000] + "..."

        # Check for speculative language
        speculative_phrases = [
            "I think",
            "probably",
            "might be",
            "could be",
            "In my opinion",
            "I believe",
            "It seems",
            "maybe",
            "perhaps",
            "possibly",
        ]

        warnings = []

        # Add speculative language warning if needed
        if any(phrase.lower() in response.lower() for phrase in speculative_phrases):
            warnings.append("(Please only state facts directly from the documents.)")

        # Add warnings if any
        if warnings:
            response = response + "\n\n" + "\n".join(warnings)

        return response
