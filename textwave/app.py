from flask import Flask, request, jsonify
import os
import glob
import time
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
) # Set up logging for debugging and monitoring.

from modules.extraction.embedding import Embedding
from modules.retrieval.index.bruteforce import BruteForceIndex
from modules.retrieval.index.hnsw import HNSWIndex
from modules.retrieval.index.lsh import LSHIndex
from modules.retrieval.search import Search
from modules.retrieval.reranker import Reranker
from modules.extraction.preprocessing import DocumentProcessing

app = Flask(__name__) # Initialize Flask app.

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Default system parameters.
CORPUS_GLOB_PATTERN = os.path.join(BASE_DIR, "storage", "*.txt.clean")

CHUNKING_STRATEGY = "fixed-length"
CHUNKING_PARAMETERS = {"chunk_size": 10, "overlap_size": 0}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_STRATEGY = "bruteforce"
INDEX_PARAMETERS = {}
RERANKING_STRATEGY = "cross_encoder"
TOP_K = 5

index = None # Global model/index/searcher references.
searcher = None
embedding_model = None
reranker = None


def initialize_index():
    """
    Initializes the embedding model, reranker, document processing,
    and builds the search index from text files in the storage directory.

    Returns:
        index (BaseIndex): The constructed retrieval index object.

    Raises:
        RuntimeError: If no corpus files are found.
        Exception: For other failures during initialization.
    """
    global index, searcher, embedding_model, reranker

    try:
        logging.info("Initializing embedding model...")
        embedding_model = Embedding(EMBEDDING_MODEL)
        reranker = Reranker(RERANKING_STRATEGY)
        logging.info("Embedding model and reranker initialized.")

        documents = []
        metadata = []

        files = glob.glob(CORPUS_GLOB_PATTERN) # Locate cleaned corpus files.

        if not files:
            logging.error(f"No corpus files found with pattern: {CORPUS_GLOB_PATTERN}")
            raise RuntimeError("No corpus files found.")

        logging.info(f"Found {len(files)} files to process.")
        
        for filepath in files: # Process and chunk each file.
            filename = os.path.basename(filepath)
            logging.debug(f"Chunking file: {filename}")
            processor = DocumentProcessing()

            try:
                start_chunk = time.time()
                chunks = processor.chunk_text(
                    filepath,
                    strategy=CHUNKING_STRATEGY,
                    chunk_size=CHUNKING_PARAMETERS["chunk_size"],
                    overlap_size=CHUNKING_PARAMETERS["overlap_size"],
                )
                chunk_time = time.time() - start_chunk
                logging.debug(f"Chunked {len(chunks)} chunks from {filename} in {chunk_time:.2f}s")
            except Exception as e:
                logging.error(f"Failed to chunk {filename}: {str(e)}")
                continue

            documents.extend(chunks)
            metadata.extend([{"source": filename}] * len(chunks))

        logging.info(f"Total chunks collected: {len(documents)}")

        dim = 384 # Assume 384-dimensional embeddings (for MiniLM model).
        
        if INDEX_STRATEGY == "bruteforce": # Choose indexing strategy.
            index = BruteForceIndex(dim=dim)
        elif INDEX_STRATEGY == "hnsw":
            index = HNSWIndex(dim=dim)
        elif INDEX_STRATEGY == "lsh":
            index = LSHIndex(dim=dim)
        else:
            raise NotImplementedError(f"Index strategy '{INDEX_STRATEGY}' not implemented.")

        index.set_embedder(embedding_model.encode) # Set embedder function and build index.
        logging.info("Building index...")

        index.build(documents, metadata)
        searcher = Search(index) # Wrap index with search utility.
        logging.info("Index and searcher ready.")

        return index

    except Exception as e:
        logging.error(f"Failed to initialize index: {str(e)}")
        raise

@app.route("/generate", methods=["POST"])
def generate_answer():
    """
    Flask route that handles POST requests with a text query and returns
    a simulated answer generated from the most relevant document chunks.

    Request JSON:
        {
            "query": "your question here"
        }

    Returns:
        flask.Response: JSON response with the query and generated answer.
        HTTP 400 if query is missing.
        HTTP 500 if search or reranking fails.
    """
    start_time = time.time()
    global index, searcher, embedding_model, reranker

    data = request.get_json(force=True)
    query = data.get("query", "").strip()

    if not query:

        return jsonify({"error": "No query provided"}), 400

    logging.info(f"Received query: {query}")
    
    if index is None or searcher is None or embedding_model is None or reranker is None: # Lazy initialization if not already set up.
        logging.info("Index not yet initialized, starting now...")
        init_start = time.time()

        initialize_index()
        logging.info(f"Index initialized in {time.time() - init_start:.2f}s")

    logging.info("Searching for top-k documents...")

    search_start = time.time()

    try: # Retrieve top-k documents.
        results = searcher.search(query, TOP_K)
        candidate_chunks = [res['text'] for res in results]
    except Exception as e:
        logging.error(f"Search failed: {str(e)}")

        return jsonify({"error": "Search failed"}), 500
    
    logging.info(f"Retrieved {len(candidate_chunks)} chunks in {time.time() - search_start:.2f}s")
    logging.info(f"Reranking using strategy: {RERANKING_STRATEGY}")

    rerank_start = time.time() # Rerank results using chosen strategy.

    try:
        if RERANKING_STRATEGY == "cross_encoder":
            reranked_chunks, _, _ = reranker.cross_encoder_rerank(query, candidate_chunks)
        elif RERANKING_STRATEGY == "tfidf":
            reranked_chunks, _, _ = reranker.tfidf_rerank(query, candidate_chunks)
        elif RERANKING_STRATEGY == "bow":
            reranked_chunks, _, _ = reranker.bow_rerank(query, candidate_chunks)
        elif RERANKING_STRATEGY == "sequential":
            seq_k1 = 5
            seq_k2 = 3
            reranked_chunks, _, _ = reranker.sequential_rerank(query, candidate_chunks, seq_k1, seq_k2)
        else:
            reranked_chunks = candidate_chunks
    except Exception as e:
        logging.error(f"Reranking failed: {str(e)}")
        reranked_chunks = candidate_chunks

    logging.info(f"Reranking completed in {time.time() - rerank_start:.2f}s")

    context = " ".join(reranked_chunks[:3])
    answer = f"Simulated answer based on: {context[:300]}..." # Simulate answer from top reranked chunks.

    total_time = time.time() - start_time
    logging.info(f"Request completed in {total_time:.2f}s")

    return jsonify({"query": query, "answer": answer})

if __name__ == "__main__":
    logging.info("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)