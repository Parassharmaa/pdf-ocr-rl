"""Model loading and LoRA configuration for Qwen2.5-VL."""


def load_model_for_training(
    model_name: str = "unsloth/Qwen2.5-VL-3B-Instruct",
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    load_in_4bit: bool = True,
):
    """Load model with Unsloth for GRPO training.

    Returns (model, tokenizer) tuple.
    """
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def load_model_for_inference(
    model_path: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """Load a fine-tuned model for inference."""
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    FastVisionModel.for_inference(model)
    return model, tokenizer


def load_base_model_for_inference(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    max_seq_length: int = 2048,
):
    """Load the base (non-fine-tuned) model for comparison."""
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    FastVisionModel.for_inference(model)
    return model, tokenizer
