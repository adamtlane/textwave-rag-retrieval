from typing import List, Dict, Any

class Search:
    """
    Generic search interface for any index implementing `search(query_text, top_k)`.

    Example usage:
        searcher = Search(index_instance)
        results = searcher.search("your query", top_k=5)
        # results: List of dicts with keys: 'doc_id', 'text', 'score'
    """

    def __init__(self, index):
        """
        Args:
            index: An index instance with a .search(query_text, top_k) method.
        """
        self.index = index

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a search over the index using the input query string.

        Args:
            query_text (str): The query string to search for.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict]: List of results containing 'doc_id', 'text', and 'score'.
        """
        return self.index.search(query_text, top_k)

if __name__ == "__main__": # Optional example standalone test (only run if invoked directly).

    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer('all-MiniLM-L6-v2') # Dummy embedder.
    embed_fn = lambda texts: model.encode(texts, convert_to_numpy=True)

    class DummyIndex: # Example dummy index class mimicking the real one.

        def __init__(self):
            self.corpus = ["The quick brown fox", "Jumps over the lazy dog", "Hello world"]
            self.doc_ids = ["doc1", "doc2", "doc3"]
            self.embedder = embed_fn
            embeddings = self.embedder(self.corpus)
            self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        def search(self, query_text, top_k=3):
            query_emb = self.embedder([query_text])
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            scores = np.dot(self.embeddings, query_emb.T).flatten()
            top_indices = scores.argsort()[::-1][:top_k]
            results = []

            for idx in top_indices:
                results.append({
                    'doc_id': self.doc_ids[idx],
                    'text': self.corpus[idx],
                    'score': float(scores[idx])
                })
                
            return results

    dummy_index = DummyIndex() # Instantiate Search with dummy index.
    searcher = Search(dummy_index)

    query = "quick fox"
    print(f"Searching for: {query}")
    results = searcher.search(query, top_k=2)

    for res in results:
        print(res)