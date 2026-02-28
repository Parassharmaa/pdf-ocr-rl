# Literature Review: RL Fine-Tuning for PDF-to-Markdown OCR

## 1. Problem Statement

PDF-to-markdown conversion requires accurately extracting text, tables, code blocks, headings, lists, and other structured content from document images. While large VLMs (GPT-4o, Claude) handle this well, smaller models (3-4B params) struggle with layout fidelity and hallucination, especially with multi-language content.

## 2. Base Models for Document OCR

### 2.1 Qwen2.5-VL Family (Jan 2025)
- **3B**: Edge-optimized, outperforms previous Qwen2-VL-7B on many tasks
- **7B**: Outperforms GPT-4o-mini; most GRPO training examples use this size
- Multi-language OCR, 32K context, dynamic resolution support
- Reference: [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)

### 2.2 Qwen3-VL Family (Oct 2025)
- **2B/4B dense** and **30B-A3B MoE** variants
- 32 OCR languages (up from 19), 256K context
- Better low-light, blur, tilt, rare character handling
- Reference: [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)

### 2.3 Specialized OCR Models
- **Nanonets OCR 2**: Fine-tuned Qwen2.5-VL-3B on 3M+ document pages
- **OCRFlux-3B**: Qwen2.5-VL-3B fine-tuned for OCR on consumer GPUs
- **Dots.ocr (1.7B)**: SOTA multilingual document parsing to markdown
- **GLM-OCR (0.9B)**: CJK support with structure-first output

### 2.4 Selected Model: Qwen2.5-VL-3B-Instruct
Chosen for: best GRPO ecosystem support, proven OCR capabilities, fits in 24GB VRAM with 4-bit quantization, Unsloth pre-quantized model available.

## 3. GRPO (Group Relative Policy Optimization)

### 3.1 Algorithm
Introduced in DeepSeekMath ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)):
- Eliminates critic/value model (saves ~50% memory vs PPO)
- Generates G completions per prompt
- Advantages computed by group normalization: `adv_i = (r_i - mean(r)) / std(r)`
- KL penalty against reference policy prevents collapse
- Every token in a completion shares the same advantage (outcome-level)

### 3.2 GSPO Variant
- "Group Sequence Policy Optimization" supported by Unsloth
- Adds importance sampling at sequence level (`importance_sampling_level="sequence"`)
- Uses DR-GRPO loss (`loss_type="dr_grpo"`) for more stable training

## 4. Reward Functions for Document OCR

### 4.1 olmOCR 2: Unit Test Rewards
- Source: [arXiv:2510.19817](https://arxiv.org/abs/2510.19817)
- Generates synthetic unit tests per document (e.g., "does output contain equation x^2?")
- Binary pass/fail per test; page reward = fraction passed
- 28 completions per document, 2,186 training pages
- Additional rewards for correct EOS token and metadata

### 4.2 Infinity-Parser: Multi-Aspect Reward
- Source: [arXiv:2506.03197](https://arxiv.org/abs/2506.03197)
- Normalized edit distance between predicted and ground-truth markdown
- Paragraph count accuracy (structural fidelity)
- Reading order preservation
- 55K training documents (synthetic + real)

### 4.3 NuMarkdown: Layout-Centric Reward
- Source: [HF: numind/NuMarkdown-8B-Thinking](https://huggingface.co/numind/NuMarkdown-8B-Thinking)
- Two-phase: SFT first, then GRPO
- Layout-centric reward prioritizing faithful formatting
- First "reasoning OCR" model (thinking tokens before markdown)

### 4.4 Our Approach: Composite Reward
Combining sub-rewards:
1. **Edit distance** (40%): Levenshtein similarity ratio
2. **Reading order** (25%): Content blocks in correct sequence
3. **Heading accuracy** (20%): Heading count, level, and text matching
4. **Structural validity** (15%): Balanced code fences, tables, formatting

## 5. Training Infrastructure

### 5.1 Unsloth
- Vision GRPO support for Qwen2.5-VL and Qwen3-VL
- 90% less VRAM, 1.5-2x faster than standard FA2
- 4-bit quantization with QLoRA
- vLLM integration for fast GRPO rollouts

### 5.2 GPU Requirements
- Qwen2.5-VL-3B + 4-bit + GRPO: ~12-16GB VRAM with Unsloth
- RTX 3090 (24GB) at $0.22/hr on RunPod community cloud
- Estimated training: 4-8 hours for 200 GRPO steps

## 6. Dataset Strategy

### 6.1 Sources
- English: GitHub repos (React, TypeScript, Go, Rust, Python, K8s, Docker, Node, Vue, Svelte)
- Japanese: GitHub JA translations, Wikipedia articles, synthetic documents
- Total: ~60 markdown sources → ~500 image-markdown pairs

### 6.2 Rendering Pipeline
- Markdown → HTML (Python markdown library) → PDF (WeasyPrint) → PNG (PyMuPDF)
- Variation: multiple font sizes, margins, font families
- Japanese: Noto CJK fonts for proper rendering

## 7. Key Takeaways

1. GRPO is the dominant RL approach for document OCR (no critic model, efficient)
2. Composite rewards outperform single-metric rewards for structured output
3. QLoRA (4-bit) makes 3B model training feasible on consumer GPUs
4. The SFT→GRPO two-phase approach yields best results but adds cost
5. Budget of $10 is sufficient for a meaningful training run on RTX 3090
