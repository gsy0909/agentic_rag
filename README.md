# OmniSearch: Agentic RAG Framework

OmniSearch is a sophisticated Agentic RAG framework designed for complex information retrieval tasks. It features multi-modal retrieval, a dynamic concept tree (ontology), and an agentic search flow with query planning, verification, and reflection.

## Key Features

- **Multi-modal Retrieval**: Combines BM25, HNSW vector search, Spacy entity recognition, and Concept Tree bonus scoring.
- **Dynamic Concept Tree**: Automatically builds and refines a hierarchical taxonomy of the corpus using LLM-guided clustering and splitting.
- **Agentic Search Flow**:
  - **Query Planner**: Decomposes complex queries into atomic sub-queries.
  - **Verifier**: Extracts atomic facts and maintains an evidence chain.
  - **Reflector**: Audits the progress and decides if further search turns are needed.
- **Modular Design**: Clean separation of concerns with dedicated modules for models, indexing, search, and evaluation.
- **Parameter Management**: Uses Hydra and YAML for flexible configuration.

## Project Structure

- `src/core/`: Main engine logic.
- `src/models/`: Unified interface for local and remote models.
- `src/indexing/`: Index construction and management.
- `src/search/`: Retrieval and ranking logic.
- `src/planner/`: Query decomposition.
- `src/verifier/`: Evidence verification and reflection.
- `src/prompts/`: Centralized prompt management.
- `src/io/`: Data loading and result saving.
- `src/evaluation/`: Performance metrics.
- `config/`: Configuration files.

## Usage

source /root/autodl-tmp/agentic_rag/.venv/bin/activate

1. Set your API key:
   ```bash
   export QWEN_API_KEY=your_api_key
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

streamlit run app.py

## Configuration

Modify `config/config.yaml` and the files in `config/` to adjust model parameters, search settings, and dataset paths.

