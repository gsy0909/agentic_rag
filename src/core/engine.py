import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm_asyncio

from ..models.remote_model import RemoteLLMClient
from ..models.local_model import QwenEmbeddingModel
from ..planner.query_planner import QueryPlanner
from ..indexing.bm25_index import BM25Index
from ..indexing.vector_index import VectorIndex
from ..indexing.entity_index import EntityIndex
from ..indexing.ontology_index import OntologyIndex
from ..search.retriever import MultiModalRetriever
from ..search.ranker import HybridRanker
from ..verifier.verifier import Verifier
from ..verifier.reflector import Reflector
from ..utils.logger import setup_logger
from ..io.data_loader import DataLoader
from .resources import ResourceManager

logger = setup_logger(__name__)

class OmniSearch:
    """
    OmniSearch: An Agentic RAG Engine with Multi-modal Retrieval and Concept Tree.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize Resource Manager
        ResourceManager.setup(config.get('resources', {}))
        
        self.corpus = {} # Map of doc_id -> text
        self.storage_dir = config['storage']['path']
        
        # Initialize Models
        # concurrency settings
        llm_conc = config['search'].get('llm_concurrency', 4) if 'search' in config else 4
        emb_conc = config['search'].get('embedding_concurrency', 1) if 'search' in config else 1
        self.subquery_concurrency = config['search'].get('subquery_concurrency', 4) if 'search' in config else 4

        self.llm_client = RemoteLLMClient(
            api_key=config['model']['api_key'],
            base_url=config['model']['base_url'],
            model=config['model']['llm_name'],
            api_mode=config['model'].get('api_mode', 'auto'),
            max_concurrency=llm_conc
        )
        self.embedding_model = QwenEmbeddingModel(
            model_name=config['model']['embedding_name'],
            max_concurrency=emb_conc,
            target_dim=config['model'].get('embedding_dim')
        )
        
        # Initialize Indices
        storage_dir = config['storage']['path']
        os.makedirs(storage_dir, exist_ok=True)
        self.bm25_index = BM25Index(os.path.join(storage_dir, "bm25"))
        self.vector_index = VectorIndex(
            os.path.join(storage_dir, "vector"), 
            dimension=config['model']['embedding_dim'],
            hnsw_m=config['search'].get('faiss', {}).get('hnsw_m', 32),
            ef_construction=config['search'].get('faiss', {}).get('ef_construction', 200)
        )
        self.entity_index = EntityIndex(
            os.path.join(storage_dir, "entity"),
            spacy_model=config['search'].get('spacy', {}).get('model', 'en_core_web_lg')
        )
        self.ontology_index = OntologyIndex(
            os.path.join(storage_dir, "ontology.json"),
            config=config['search'].get('ontology', {})
        )
        # Whether to enable ontology-based search fusion (default: False)
        self.ontology_enabled = bool(config['search'].get('ontology_enabled', False)) if 'search' in config else False
        
        # Initialize Components
        self.planner = QueryPlanner(self.llm_client)
        self.retriever = MultiModalRetriever(
            self.bm25_index, self.vector_index, self.entity_index, self.ontology_index,
            self.embedding_model, self.llm_client
        )
        self.ranker = HybridRanker(
            weights=config['search'].get('weights'), 
            concept_bonus=config['search'].get('concept_bonus', 1.5),
            rrf_k=config['search'].get('rrf_k', 60)
        )
        self.verifier = Verifier(self.llm_client)
        self.reflector = Reflector(self.llm_client)

        # --- Compression module (plug-and-play) ---
        # Disabled by default. Enable via config: compression.enabled: true
        # When disabled, this block is fully skipped and the pipeline is unchanged.
        compression_cfg = config.get('search', {}).get('compression', {})
        self.compression_enabled = bool(compression_cfg.get('enabled', False))
        if self.compression_enabled:
            from ..compression.compressor import ContextCompressor
            self.compressor = ContextCompressor(self.embedding_model, compression_cfg)

        self.max_turns = config['search'].get('max_turns', 3)

    async def build_indices(self, corpus: List[Dict[str, Any]]):
        """Build all indices if they don't exist."""
        self.corpus = {str(doc['id']): doc['text'] for doc in corpus}
        # Build indices in threadpool where operations are blocking
        await asyncio.to_thread(self.bm25_index.build, corpus)
        await asyncio.to_thread(self.vector_index.build, corpus, self.embedding_model)
        await asyncio.to_thread(self.entity_index.build, corpus)

        # Try to reuse precomputed embeddings from vector index to avoid recomputing
        doc_vecs = self.vector_index.load_embeddings()
        if doc_vecs is None:
            # fallback: compute via embedding model
            encode_fn = getattr(self.embedding_model, "encode_async", None)
            texts = [doc["text"] for doc in corpus]
            if encode_fn is None:
                doc_vecs = await asyncio.to_thread(self.embedding_model.encode, texts)
            else:
                doc_vecs = await encode_fn(texts)

        if not self.ontology_enabled:
            logger.info("Ontology search is disabled; skipping ontology index build.")
            return

        force_rebuild_ontology = bool(self.config.get('storage', {}).get('force_rebuild_ontology', False))
        await self.ontology_index.build(
            corpus,
            self.llm_client,
            self.embedding_model,
            doc_vecs=doc_vecs,
            force=force_rebuild_ontology,
        )

    def update_search_params(
        self,
        max_turns: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
        concept_bonus: Optional[float] = None,
        rrf_k: Optional[int] = None,
        ontology_enabled: Optional[bool] = None,
        compression_enabled: Optional[bool] = None,
    ) -> None:
        """Update search parameters at runtime (for UI use). Does not affect index or model."""
        from ..search.ranker import HybridRanker
        if max_turns is not None:
            self.max_turns = max_turns
        if ontology_enabled is not None:
            self.ontology_enabled = ontology_enabled
        if compression_enabled is not None:
            self.compression_enabled = compression_enabled
        if any(x is not None for x in [weights, concept_bonus, rrf_k]):
            self.ranker = HybridRanker(
                weights=weights if weights is not None else self.ranker.weights,
                concept_bonus=concept_bonus if concept_bonus is not None else self.ranker.concept_bonus,
                rrf_k=rrf_k if rrf_k is not None else self.ranker.rrf_k,
            )

    async def search(self, query: str, request_id: Optional[str] = None, trace_mode: bool = False) -> Dict[str, Any]:
        """Perform agentic multi-turn search.

        Args:
            trace_mode: When True (UI use), collects enriched per-subquery trace data
                        (retrieval counts, ranked docs, compression stats) and returns it
                        as ``turn_details`` in the result dict. Has zero effect on JSONL
                        output files or any other behavior when False (default).
        """
        current_query = query
        turn = 0
        all_evidences = []
        all_doc_ids = set()
        trace_queries: List[str] = []
        turn_trace_details: List[Dict[str, Any]] = []
        compression_log: List[Dict[str, Any]] = []  # collects per-doc compression stats
        ui_turn_details: List[Dict[str, Any]] = []  # enriched trace, only populated when trace_mode=True

        while turn < self.max_turns:
            logger.info(f"Turn {turn + 1}: Planning for query: {current_query}")
            plan = await self.planner.plan(current_query)
            sub_queries = plan.get("sub_queries", [current_query])
            # record the query used in this turn
            trace_queries.append(current_query)
            # Process sub-queries with controlled concurrency
            sem = asyncio.Semaphore(self.subquery_concurrency)

            last_reflection = None

            async def _process_subq(sub_q: str):
                # limit number of parallel retrieve calls
                async with sem:
                    multi_results = await self.retriever.retrieve(sub_q, use_ontology=self.ontology_enabled)

                ranked_docs = self.ranker.rank(multi_results)

                # Fetch full text for verification
                docs_for_verify = []
                for doc_res in ranked_docs:
                    doc_id = str(doc_res['id'])
                    if doc_id in self.corpus:
                        docs_for_verify.append({"id": doc_id, "text": self.corpus[doc_id]})

                # --- Plug-and-play compression ---
                # When compression_enabled=False this block is skipped entirely;
                # the original [:8000] truncation inside verifier.py acts as fallback.
                subq_compression: List[Dict[str, Any]] = []
                if self.compression_enabled:
                    compressed_results = await asyncio.gather(*[
                        self.compressor.compress_doc(sub_q, d["id"], d["text"])
                        for d in docs_for_verify
                    ])
                    for r in compressed_results:
                        compression_log.append({
                            "turn": turn + 1,
                            "sub_query": sub_q,
                            "doc_id": r["doc_id"],
                            "orig_words": r["orig_words"],
                            "comp_words": r["comp_words"],
                            "ratio": r["ratio"],
                        })
                        if trace_mode:
                            subq_compression.append({
                                "doc_id": r["doc_id"],
                                "orig_words": r["orig_words"],
                                "comp_words": r["comp_words"],
                                "ratio": r["ratio"],
                            })
                    docs_for_verify = [
                        {"id": r["doc_id"], "text": r["compressed_text"]}
                        for r in compressed_results
                        if r["compressed_text"].strip()
                    ]

                verification = await self.verifier.verify(sub_q, docs_for_verify)
                verification["sub_query"] = sub_q

                if trace_mode:
                    return {
                        "verification": verification,
                        "retrieval_counts": {
                            "bm25": len(multi_results.get("bm25", [])),
                            "vector": len(multi_results.get("vector", [])),
                            "entity": len(multi_results.get("entity", [])),
                            "ontology": len(multi_results.get("ontology", [])),
                        },
                        "ranked_top10": [
                            {
                                "id": str(d["id"]),
                                "score": round(float(d["score"]), 4),
                                "text_preview": self.corpus.get(str(d["id"]), "")[:300],
                            }
                            for d in ranked_docs[:10]
                        ],
                        "compression_stats": subq_compression,
                    }
                return verification

            tasks = [_process_subq(sq) for sq in sub_queries]
            turn_results = await asyncio.gather(*tasks)

            # When trace_mode=True, turn_results contains enriched dicts; extract verifications.
            # When trace_mode=False, turn_results is the original list of verification dicts.
            if trace_mode:
                verifications = [r["verification"] for r in turn_results]
                ui_turn_details.append({
                    "turn": turn + 1,
                    "main_query": current_query,
                    "plan": {
                        "sub_queries": sub_queries,
                        "intent": plan.get("intent", ""),
                        "rewritten_query": plan.get("rewritten_query", current_query),
                    },
                    "sub_queries": [
                        {
                            "sub_query": r["verification"].get("sub_query", ""),
                            "retrieval_counts": r["retrieval_counts"],
                            "ranked_top10": r["ranked_top10"],
                            "compression_stats": r["compression_stats"],
                            "verifier_result": r["verification"],
                        }
                        for r in turn_results
                    ],
                })
            else:
                verifications = turn_results

            # Original turn_trace_details format — unchanged regardless of trace_mode
            turn_trace_details.append({
                "turn": turn + 1,
                "main_query": current_query,
                "sub_queries": [
                    {
                        "sub_query": v.get("sub_query", ""),
                        "verifier_result": v,
                    }
                    for v in verifications
                ],
            })

            for verification in verifications:
                all_evidences.extend(verification.get("evidences_chain", []))
                all_doc_ids.update(verification.get("keep_ids", []))

            # Reflect on the current turn's results using the query that was planned/executed this turn
            # Provide original query, current (rewritten) query, verifier results and accumulated evidences
            reflection = await self.reflector.reflect(query, current_query, verifications, all_evidences)
            last_reflection = reflection
            if trace_mode:
                ui_turn_details[-1]["reflection"] = {
                    "answered": reflection.get("answered", False),
                    "thought": reflection.get("thought", ""),
                    "new_query": reflection.get("new_query", ""),
                }
            if reflection.get("answered"):
                logger.info("Query fully answered.")
                # persist trace for answered requests as well
                try:
                    self._persist_query_traces(request_id, trace_queries, turn_trace_details, compression_log)
                except Exception:
                    logger.warning("Failed to write query trace for answered request")

                result = {
                    "query": query,
                    "answer": reflection.get("final_answer"),
                    "evidences": all_evidences,
                    "thought": reflection.get("thought"),
                    "doc_ids": list(all_doc_ids),
                    "turns": turn + 1,
                }
                if trace_mode:
                    result["turn_details"] = ui_turn_details
                return result
            # Update current_query to the new query suggested by reflector (defaults to previous current_query)
            current_query = reflection.get("new_query", current_query)
            turn += 1

        forced_final = await self.reflector.force_answer(
            query,
            current_query,
            verifications if 'verifications' in locals() else [],
            all_evidences,
        )
        final_ans = (
            forced_final.get("final_answer")
            or (last_reflection.get("final_answer") if last_reflection else None)
            or "Unknown"
        )
        final_thought = (
            forced_final.get("thought")
            or (last_reflection.get("thought") if last_reflection else None)
        )
        # persist trace: one line per request with id first
        try:
            self._persist_query_traces(request_id, trace_queries, turn_trace_details, compression_log)
        except Exception:
            logger.warning("Failed to write query trace")
        result = {
            "query": query,
            "answer": final_ans,
            "evidences": all_evidences,
            "thought": final_thought,
            "doc_ids": list(all_doc_ids),
            "turns": turn,
        }
        if trace_mode:
            result["turn_details"] = ui_turn_details
        return result

    def _persist_query_traces(
        self,
        request_id: Optional[str],
        trace_queries: List[str],
        turn_trace_details: List[Dict[str, Any]],
        compression_log: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        os.makedirs(self.storage_dir, exist_ok=True)
        rid = str(request_id) if request_id is not None else "unknown"

        trace_path = os.path.join(self.storage_dir, "query_traces.jsonl")
        with open(trace_path, "a", encoding="utf-8") as tf:
            ordered = {"id": rid, "queries": trace_queries}
            tf.write(json.dumps(ordered, ensure_ascii=False) + "\n")

        detail_trace_path = os.path.join(self.storage_dir, "query_turn_details.jsonl")
        with open(detail_trace_path, "a", encoding="utf-8") as tf:
            ordered = {"id": rid, "turns": turn_trace_details}
            tf.write(json.dumps(ordered, ensure_ascii=False) + "\n")

        if compression_log:
            comp_path = os.path.join(self.storage_dir, "compression_stats.jsonl")
            with open(comp_path, "a", encoding="utf-8") as cf:
                for entry in compression_log:
                    cf.write(json.dumps({"request_id": rid, **entry}, ensure_ascii=False) + "\n")
