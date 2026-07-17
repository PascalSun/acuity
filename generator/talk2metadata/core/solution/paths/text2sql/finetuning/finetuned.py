"""Finetuning mode for Text2SQL using local models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

from ..direct_retriever import (
    DirectText2SQLRetriever,
)

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Mock LLM response object."""

    content: str


class LocalLLMWrapper:
    """Wrapper for local HuggingFace/PEFT models to match LLMProvider interface."""

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        device: str = "auto",
        quantization: Optional[str] = "4bit",  # 4bit, 8bit, or None
        trust_remote_code: bool = True,
    ):
        """Initialize local LLM wrapper.

        Args:
            model_path: Path to base model or HF Hub ID
            adapter_path: Path to LoRA adapter or HF Hub ID
            device: Device to use (auto, cuda, mps, cpu)
            quantization: Quantization mode (4bit, 8bit, or None)
            trust_remote_code: Whether to trust remote code
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device = self._get_device(device)
        self.quantization = quantization

        logger.info(
            f"Initializing LocalLLMWrapper with model={model_path}, device={self.device}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = self._load_model(
            model_path, adapter_path, self.device, quantization, trust_remote_code
        )

        logger.info("Local model loaded successfully")

    def _get_device(self, device: str) -> str:
        """Get appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_model(
        self,
        model_path: str,
        adapter_path: Optional[str],
        device: str,
        quantization: Optional[str],
        trust_remote_code: bool,
    ) -> Any:
        """Load model and adapter."""

        # Configure quantization
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        load_kwargs = {
            "device_map": "auto" if device != "cpu" else None,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
        }

        if (
            bnb_config and device != "mps"
        ):  # MPS doesn't support bitsandbytes yet typically
            load_kwargs["quantization_config"] = bnb_config

        logger.info(f"Loading base model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        if device == "mps" and not bnb_config:
            model = model.to("mps")

        if adapter_path:
            logger.info(f"Loading adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)

        return model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """Generate text from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation args

        Returns:
            Likely LLMResponse with 'content' attribute
        """
        # Construct full prompt
        full_prompt = prompt
        if system_prompt:
            # Simple formatting - standard chat templates are better but this is a fallback
            # Ideally we should use tokenizer.apply_chat_template if available
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                # Check if tokenizer has chat template
                if getattr(self.tokenizer, "chat_template", None):
                    full_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # Fallback
                    full_prompt = (
                        f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                    )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=(
                    temperature if temperature > 0 else 0.01
                ),  # Avoid 0 exactly if problematic
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return LLMResponse(content=generated_text)


class FinetunedRetriever(DirectText2SQLRetriever):
    """Retriever using local fine-tuned model for Text2SQL."""

    def __init__(
        self,
        schema_metadata: SchemaMetadata,
        **kwargs: Any,
    ):
        """Initialize finetuned retriever.

        Args:
            schema_metadata: Schema metadata
            **kwargs: Additional configuration
        """
        # Call grand-parent init (BaseRetriever) indirectly?
        # Actually DirectText2SQLRetriever calls BaseText2SQLRetriever.__init__
        # We need to bypass the default LLM initialization in BaseText2SQLRetriever
        # or overwrite it immediately after.

        # To avoid BaseText2SQLRetriever trying to load OpenAI/etc keys, we might need to trick it
        # However, BaseText2SQLRetriever.__init__ requires existing config setup for agent.

        # Better approach: Initialize manually without calling super().__init__ completely,
        # or let super().__init__ fail/succeed and then replace self.llm.

        # But BaseText2SQLRetriever sets up self.engine as well.
        # Let's let super().__init__ run, but we suppress potential LLM init errors or just overwrite self.llm

        # Actually, BaseText2SQLRetriever initializes self.llm using config['agent'].
        # If agent is disabled or configured for local, it might be weird.

        # Let's just run super().__init__. If it fails due to missing keys, that's an issue.
        # But we can override the behavior by configuring a 'dummy' provider in config temporarily?
        # Or we can just reimplement the init parts we need.

        # Re-implementing parts of __init__ to avoid dependency on global agent config
        self.schema_metadata = schema_metadata
        self.target_table = schema_metadata.target_table

        # 1. Database Connection (copied from BaseText2SQLRetriever)
        connection_string = kwargs.get("connection_string")
        engine = kwargs.get("engine")

        config = get_config()
        if engine:
            self.engine = engine
            self._own_engine = False
        elif connection_string:
            from sqlalchemy import create_engine

            self.engine = create_engine(connection_string)
            self._own_engine = True
        else:
            # Try to get from config
            ingest_config = config.get("ingest", {})
            source_path = ingest_config.get("source_path")
            data_type = ingest_config.get("data_type", "csv")

            if data_type in ("database", "db") and source_path:
                from sqlalchemy import create_engine

                logger.info(f"Using connection string from config: {source_path}")
                self.engine = create_engine(source_path)
                self._own_engine = True
            else:
                # It might be fine if we are just testing, but generally we need DB
                # If we don't have DB, we can't execute SQL, so BaseText2SQLRetriever won't work well
                # But let's assume valid config.
                # If missing, try to initialize standard way just in case
                pass

        # 2. Initialize Local LLM
        modes_cfg = config.get("modes", {})
        if isinstance(modes_cfg, dict):
            mode_block = modes_cfg.get("text2sql.finetuning", {})
        else:
            mode_block = {}

        mode_config = (
            mode_block.get("retriever", {}) if isinstance(mode_block, dict) else {}
        )

        model_path = (
            mode_block.get("model_path") if isinstance(mode_block, dict) else None
        )
        adapter_path = (
            mode_block.get("adapter_path") if isinstance(mode_block, dict) else None
        )
        device = (
            mode_block.get("device", "auto") if isinstance(mode_block, dict) else "auto"
        )

        if not model_path:
            # Fallback or error
            logger.warning(
                "No model_path configured for text2sql.finetuning mode. Using default."
            )
            model_path = (
                "QuantFactory/Mistral-7B-Instruct-v0.3-GGUF"  # Placeholder default
            )

        self.llm = LocalLLMWrapper(
            model_path=model_path, adapter_path=adapter_path, device=device
        )

        # 3. Context
        self.context = mode_config.get("context", "").strip()

        logger.info(f"FinetunedRetriever initialized with model: {model_path}")
