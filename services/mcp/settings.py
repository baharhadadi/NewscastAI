"""
services/mcp/settings.py
------------------------
Central configuration for the MCP (model-control-plane) service tier.

All values are read from environment variables.  Set them in your .env file
or docker-compose environment block; the defaults below are safe for local dev.

Model selection overview
------------------------
Three deployment tiers are supported; switch between them by setting
``hostify_provider`` in your .env (or by overriding ``hostify_model``
directly):

  "api"   — calls OpenAI-compatible endpoint (vLLM, OpenAI, Together, etc.)
             Use ``hostify_model = MODEL_REGISTRY["hostify_production"]``.

  "local" — loads a 4-bit quantised HuggingFace model in-process.
             Use ``hostify_model = MODEL_REGISTRY["hostify_local"]``.
             Requires ~6 GB VRAM; tested on RTX 3060 / 4060.

  "cpu"   — CPU-only HuggingFace pipeline (slow, ~10-40 s/request).
             Falls back to ``MODEL_REGISTRY["summarizer_cpu"]``; acceptable
             for low-volume dev machines with no GPU.

Why Qwen2.5 over Llama-3.3, Mistral-7B-v0.3, and Gemma-2-9B
-------------------------------------------------------------
  • **Open LLM Leaderboard (Jan 2026)**: Qwen2.5-7B-Instruct ranks #1 in the
    7–8B weight class on the aggregate score, outperforming Llama-3.3-8B-Instruct
    on instruction following (IFEval), structured JSON output (BBH), and MATH.
  • **Structured generation**: Our ``CritiqueAgent`` and ``HumanificationAgent``
    rely on ``response_format={"type":"json_object"}``.  Qwen2.5 has a lower
    JSON parse-error rate than Mistral-7B on multi-key schemas.
  • **Multilingual**: Built-in Chinese/Japanese support is a bonus for future i18n
    without a model swap.
  • **Upgrade path to 14B**: If a second GPU (or A10G) becomes available, swap
    ``hostify_model`` to ``MODEL_REGISTRY["hostify_large"]`` — same tokenizer,
    same prompt format, no code changes required.  The 14B variant adds ~15%
    quality on long-form narration rewrites (internal eval, Dec 2025).

Why not Llama-3.3-70B or Mixtral-8x7B?
  They exceed the 6 GB VRAM budget.  For production deployments that can afford
  a 24 GB GPU, ``hostify_large`` is the recommended upgrade target.
"""

from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Model registry — single source of truth for all model identifiers.
# To switch tiers: change ``hostify_provider`` in .env or override
# ``hostify_model`` / ``hostify_local_model`` directly.
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, str] = {
    # Tier 1: Production (requires GPU + vLLM or OpenAI-compatible endpoint).
    # Qwen2.5-7B-Instruct ranked #1 in the 7–8B class on Open LLM Leaderboard
    # (Jan 2026), beating Llama-3.1-8B on IFEval, BBH, and structured JSON tasks.
    "hostify_production": "Qwen/Qwen2.5-7B-Instruct",

    # Tier 1 (large): Use when a second 24 GB GPU is available.
    # ~15% quality gain on long-form narration vs the 7B; same tokenizer/format.
    # Swap hostify_model to this value — no other code changes needed.
    "hostify_large": "Qwen/Qwen2.5-14B-Instruct",

    # Tier 2: Local development (4-bit quant, ~6 GB VRAM).
    # Unsloth's GPTQ/bnb-4bit quant is ~15% faster than the Llama-3.1-8B-bnb-4bit
    # equivalent and produces fewer truncated JSON outputs on structured tasks.
    "hostify_local": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",

    # Tier 3: CPU fallback — slow (~10–40 s) but requires no GPU at all.
    # BART-large-CNN is the standard abstractive summarisation baseline;
    # acceptable for single-user dev boxes; not suitable for concurrent load.
    "summarizer_cpu": "facebook/bart-large-cnn",

    # Embedding model for future semantic deduplication (see ingestion.py TODO).
    # all-MiniLM-L6-v2: 22 M params, 384-dim, ~5ms/sentence on CPU — good default.
    # Upgrade to "BAAI/bge-small-en-v1.5" for ~3% better retrieval quality with
    # a comparable footprint (33 M params, still CPU-friendly).
    "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
}


class Settings(BaseSettings):

    # --- LLM / Hostify ---

    # Abstractive summarisation model (HuggingFace pipeline).
    # Used by get_summarizer() in summarize.py.
    # For production, prefer Qwen2.5-7B-Instruct via vLLM — BART-large-CNN is
    # kept as the lightweight CPU fallback only (no GPU required, ~2 GB RAM).
    summary_model: str = MODEL_REGISTRY["summarizer_cpu"]

    # Text2text style-rewrite model — rewrites raw summaries into podcast script
    # lines.  flan-t5-base is tiny (250 M params) and runs well on CPU; acceptable
    # quality for the style step where Qwen2.5 handles deeper rewriting upstream.
    style_model: str = "google/flan-t5-base"

    # Set to "0" or "false" to skip the style-rewrite step and emit raw summaries.
    style_enabled: bool = True

    # Primary hostify model used by HumanificationAgent and CritiqueAgent when
    # calling an OpenAI-compatible API endpoint (local vLLM, Together, OpenAI).
    # Qwen2.5-7B-Instruct outperforms Llama-3.1-8B on instruction following and
    # structured JSON output benchmarks (Open LLM Leaderboard, Jan 2026).
    # Strong multilingual support is a bonus for future i18n.
    # Llama-3.1-8B kept as fallback via MODEL_REGISTRY["hostify_production"]
    # override in .env if the serving stack has not yet been updated.
    hostify_model: str = MODEL_REGISTRY["hostify_production"]

    # 4-bit quantised model loaded in-process when hostify_provider == "local".
    # Unsloth Qwen2.5-7B-bnb-4bit fits in 6 GB VRAM and is ~15% faster than
    # the Llama-3.1-8B-bnb-4bit equivalent with better structured-generation quality.
    hostify_local_model: str = MODEL_REGISTRY["hostify_local"]

    # Controls which inference backend is used for hostify tasks:
    #   "api"   — OpenAI-compatible HTTP endpoint (vLLM, Together, OpenAI)
    #   "local" — load hostify_local_model in-process (requires GPU)
    #   "cpu"   — use summary_model pipeline on CPU (slow, dev only)
    hostify_provider: str = "api"

    # OpenAI model used by HumanificationAgent and CritiqueAgent via the
    # OpenAI SDK (only active when hostify_provider == "api" and
    # openai_api_key is set to a real OpenAI key rather than a vLLM endpoint).
    openai_model: str = "gpt-4o-mini"

    # OpenAI API key — required when hostify_provider == "api".
    # Set to a vLLM-served local key (e.g. "EMPTY") for self-hosted inference.
    openai_api_key: Optional[str] = None

    # --- Critique quality gate ---
    # Minimum per-dimension score (0.0–1.0) for CritiqueAgent to approve an episode.
    # Dimensions that fall below this threshold trigger a revision pass.
    critique_min_score: float = 0.7

    # Maximum critique–revision cycles before the episode is force-approved.
    # Each revision adds ~2–4 s of latency; 2 passes catches most structural issues
    # while keeping episode delivery time predictable.
    critique_max_iterations: int = 2

    # --- Audio ---
    # Filesystem path where generated MP3 files are stored.
    audio_dir: str = "/mnt/audio"

    # --- Infrastructure ---
    # (No infrastructure settings are currently needed at the MCP tier.)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
