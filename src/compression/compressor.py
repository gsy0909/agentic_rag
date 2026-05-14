import re
import asyncio
import numpy as np
from typing import List, Dict, Any
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation. No external dependencies."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.split()) > 3]


class EmbeddingHardCompressor:
    """
    Sentence-level hard compressor based on embedding cosine similarity.

    Inspired by the extractive compressor in RECOMP (Xu et al., ICLR 2024).
    Reuses the project's existing QwenEmbeddingModel — no additional model or
    dependency required.

    Each sentence in the document is scored against the sub-query via cosine
    similarity (embeddings are already L2-normalised, so dot-product suffices).
    Sentences below `threshold` are discarded; the remainder are concatenated
    and returned as the compressed document.
    """

    def __init__(self, embedding_model, threshold: float = 0.3, skip_short_docs: int = 150,
                 batch_size: int = 32):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.skip_short_docs = skip_short_docs
        self.batch_size = batch_size

    async def compress(self, query: str, doc_text: str) -> str:
        if len(doc_text.split()) <= self.skip_short_docs:
            return doc_text

        sentences = _split_sentences(doc_text)
        if not sentences:
            return doc_text

        try:
            # encode_async respects the GPU semaphore defined in ResourceManager
            query_emb = await self.embedding_model.encode_async([query], is_query=True)  # (1, dim)

            # Encode sentences in batches to avoid OOM on long documents
            chunks = [sentences[i:i + self.batch_size] for i in range(0, len(sentences), self.batch_size)]
            sent_embs = np.concatenate(
                [await self.embedding_model.encode_async(chunk) for chunk in chunks], axis=0
            )  # (N, dim)

            # Dot product of L2-normalised vectors == cosine similarity
            scores = np.dot(sent_embs, query_emb[0])  # (N,)

            kept = [s for s, sc in zip(sentences, scores) if sc >= self.threshold]
            if not kept:
                # Fallback: always keep the top-3 most relevant sentences
                top_idx = np.argsort(scores)[::-1][:3]
                kept = [sentences[i] for i in sorted(top_idx)]

            return " ".join(kept)
        except Exception as e:
            logger.warning(f"EmbeddingHardCompressor error, returning original doc: {e}")
            return doc_text


class CrossEncoderHardCompressor:
    """
    Sentence-level compressor using a cross-encoder reranker.

    Unlike the bi-encoder EmbeddingHardCompressor, a cross-encoder jointly
    encodes (query, sentence) pairs, capturing fine-grained interactions such
    as pronoun coreference and negation that a dual-tower model misses.

    Based on BGE-reranker-v2 (2024) / RankRAG (NeurIPS 2024).
    Requires:  pip install sentence-transformers
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 threshold: float = 0.5, skip_short_docs: int = 150):
        self.model_name = model_name
        self.threshold = threshold
        self.skip_short_docs = skip_short_docs
        self._model = None  # lazy-loaded

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"CrossEncoderHardCompressor loaded: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install it with:  pip install sentence-transformers"
                )
        return self._model

    async def compress(self, query: str, doc_text: str) -> str:
        if len(doc_text.split()) <= self.skip_short_docs:
            return doc_text
        sentences = _split_sentences(doc_text)
        if not sentences:
            return doc_text
        try:
            model = self._get_model()
            pairs = [(query, s) for s in sentences]
            scores = await asyncio.to_thread(model.predict, pairs)
            kept = [s for s, sc in zip(sentences, scores) if sc >= self.threshold]
            if not kept:
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
                kept = [sentences[i] for i in sorted(top_idx)]
            return " ".join(kept)
        except Exception as e:
            logger.warning(f"CrossEncoderHardCompressor error, returning original doc: {e}")
            return doc_text


class LLMLinguaHardCompressor:
    """
    Token-level hard compressor based on perplexity scoring.

    Based on LLMLingua (Jiang et al., EMNLP 2023) and LLMLingua-2
    (Pan et al., 2024). Uses a small BERT-based model to score each token's
    informativeness and removes low-scoring tokens up to the target
    `compression_rate`.

    Requires:  pip install llmlingua

    The underlying model is loaded lazily on first use so that the class can
    be instantiated without the package being installed (an ImportError is
    raised only when `compress()` is first called).
    """

    _HF_MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    # Max BERT token IDs per chunk. LLMLingua adds ~250 tokens of overhead
    # (instruction prefix + question + special tokens), so 200 + 250 = 450 < 512.
    _CHUNK_TOKENS = 200

    def __init__(self, compression_rate: float = 0.5, skip_short_docs: int = 150,
                 local_model_dir: str = "models/llmlingua2"):
        self.compression_rate = compression_rate
        self.skip_short_docs = skip_short_docs
        self.local_model_dir = local_model_dir
        self._compressor = None   # lazy-loaded
        self._bert_tokenizer = None  # the actual BERT tokenizer used by the model

    def _get_compressor(self):
        if self._compressor is None:
            try:
                import os
                from llmlingua import PromptCompressor

                local_config = os.path.join(self.local_model_dir, "config.json")
                if os.path.isfile(local_config):
                    model_name = self.local_model_dir
                    logger.info(f"Loading LLMLingua-2 from local path: {self.local_model_dir}")
                else:
                    model_name = self._HF_MODEL_NAME
                    logger.info(f"Local model not found, downloading from HuggingFace: {model_name}")

                self._compressor = PromptCompressor(
                    model_name=model_name,
                    use_llmlingua2=True,
                )
                # is_begin_of_new_word() in llmlingua/utils.py checks model_name to
                # determine the tokenizer scheme. A local path is unrecognized and
                # raises NotImplementedError, so always restore the canonical HF name.
                self._compressor.model_name = self._HF_MODEL_NAME

                # compressor.tokenizer may be tiktoken (OAI tokenizer) rather than
                # the BERT tokenizer the model actually uses. Load it explicitly so
                # our chunking matches the model's real token counts.
                from transformers import AutoTokenizer
                self._bert_tokenizer = AutoTokenizer.from_pretrained(model_name)

                if model_name == self._HF_MODEL_NAME:
                    try:
                        os.makedirs(self.local_model_dir, exist_ok=True)
                        self._compressor.tokenizer.save_pretrained(self.local_model_dir)
                        self._compressor.model.save_pretrained(self.local_model_dir)
                        logger.info(f"LLMLingua-2 model saved to {self.local_model_dir}")
                    except Exception as save_err:
                        logger.warning(f"Failed to save model locally: {save_err}")

                logger.info("LLMLingua-2 compressor loaded successfully.")
            except ImportError:
                raise ImportError(
                    "llmlingua is not installed. Install it with:  pip install llmlingua"
                )
        return self._compressor

    async def compress(self, query: str, doc_text: str) -> str:
        if len(doc_text.split()) <= self.skip_short_docs:
            return doc_text

        # tiktoken treats <|endoftext|>, <|im_start|> etc. as disallowed special tokens
        # and raises an error when they appear as plain text. Strip them before passing
        # to compress_prompt to avoid falling back to the original doc on every call.
        clean_text = re.sub(r'<\|[^|]+\|>', ' ', doc_text).strip()
        if not clean_text:
            return doc_text

        try:
            compressor = self._get_compressor()

            def _compress_all() -> str:
                # LLMLingua-2 prepends the question to every context window:
                # [CLS] question [SEP] context_window [SEP] ≤ 512.
                # If the sub_query itself is very long (multi-turn accumulation),
                # truncate it to leave at least 64 tokens for the context window.
                MAX_Q_TOKENS = 200
                q_ids = self._bert_tokenizer.encode(query, add_special_tokens=False)
                if len(q_ids) > MAX_Q_TOKENS:
                    safe_query = self._bert_tokenizer.decode(q_ids[:MAX_Q_TOKENS])
                    logger.debug(f"query truncated {len(q_ids)} → {MAX_Q_TOKENS} BERT tokens for LLMLingua")
                else:
                    safe_query = query
                    q_ids = q_ids  # already within limit

                safe_iter = max(16, 512 - min(len(q_ids), MAX_Q_TOKENS) - 3 - 5)
                res = compressor.compress_prompt(
                    [clean_text],
                    rate=self.compression_rate,
                    force_tokens=["\n", "?", "!", ".", ","],
                    question=safe_query,
                    iterative_size=safe_iter,
                )
                return res.get("compressed_prompt", "").strip()

            # _compress_all is CPU-bound; run in threadpool to avoid blocking the event loop
            compressed = await asyncio.to_thread(_compress_all)
            return compressed if compressed else doc_text
        except Exception as e:
            import traceback
            logger.warning(
                f"LLMLinguaHardCompressor error, returning original doc: "
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
            return doc_text


class ContextCompressor:
    """
    Plug-and-play document pre-compressor for Agentic RAG pipelines.

    Sits between the ranker and the verifier. Reduces each retrieved document
    to its query-relevant content before the verifier's LLM call, improving
    the signal-to-noise ratio of the verifier's input.

    Three interchangeable hard-compression strategies are supported:
      'embedding'     — sentence-level cosine-similarity filtering (RECOMP-style)
      'cross_encoder' — sentence-level cross-encoder reranking (BGE-reranker-v2, 2024)
      'llmlingua'     — token-level perplexity-based filtering (LLMLingua-style)

    Strategy is selected via config['hard_method']. The module is completely
    bypass-able: when compression is disabled in the engine, this class is
    never instantiated and the original pipeline runs unchanged.
    """

    def __init__(self, embedding_model, config: Dict[str, Any]):
        method = config.get("hard_method", "embedding")
        skip_short = config.get("skip_short_docs", 150)
        # Limit concurrent GPU compression calls to avoid OOM when asyncio.gather fires many at once
        self._sem = asyncio.Semaphore(config.get("gpu_concurrency", 10))

        if method == "llmlingua":
            self.hard = LLMLinguaHardCompressor(
                compression_rate=config.get("compression_rate", 0.5),
                skip_short_docs=skip_short,
                local_model_dir=config.get("local_model_dir", "models/llmlingua2"),
            )
            logger.info("ContextCompressor ready  [method: llmlingua]")
        elif method == "cross_encoder":
            self.hard = CrossEncoderHardCompressor(
                model_name=config.get("cross_encoder_model", "BAAI/bge-reranker-v2-m3"),
                threshold=config.get("similarity_threshold", 0.5),
                skip_short_docs=skip_short,
            )
            logger.info("ContextCompressor ready  [method: cross_encoder]")
        else:
            self.hard = EmbeddingHardCompressor(
                embedding_model=embedding_model,
                threshold=config.get("similarity_threshold", 0.3),
                skip_short_docs=skip_short,
                batch_size=config.get("embedding_batch_size", 32),
            )
            logger.info("ContextCompressor ready  [method: embedding]")

    async def compress_doc(self, sub_query: str, doc_id: str, doc_text: str) -> Dict[str, Any]:
        """
        Compress a single retrieved document with respect to sub_query.

        Always returns a dict so callers can log compression statistics.
        If compression fails for any reason the original text is returned
        (guaranteed no data loss).
        """
        async with self._sem:
            compressed_text = await self.hard.compress(sub_query, doc_text)
        orig_words = len(doc_text.split())
        comp_words = len(compressed_text.split()) if compressed_text else 0
        ratio = round(comp_words / max(orig_words, 1), 2)
        logger.debug(f"doc[{doc_id}] {orig_words} → {comp_words} words  (ratio={ratio:.2f})")
        return {
            "doc_id": doc_id,
            "compressed_text": compressed_text,
            "orig_words": orig_words,
            "comp_words": comp_words,
            "ratio": ratio,
        }
