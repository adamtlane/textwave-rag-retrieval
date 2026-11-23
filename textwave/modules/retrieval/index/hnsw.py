import faiss
import numpy as np
import pickle

class HNSWIndex:
    """
    FAISS HNSW index wrapper for approximate nearest neighbor search over text embeddings.

    Args:
        dim (int): Embedding dimensionality.
        embedder (callable): Function that converts list of texts to embeddings.
        metric (str): 'euclidean' or 'cosine'.
        M (int): HNSW parameter controlling number of neighbors (default=32).
        efConstruction (int): HNSW parameter controlling construction time/search accuracy (default=40).
    """

    def __init__(self, dim, embedder=None, metric='euclidean', M=32, efConstruction=40):
        self.dim = dim
        self.metric = metric.lower()
        self.embedder = embedder
        self.M = M
        self.efConstruction = efConstruction

        if self.metric in ['euclidean', 'minkowski']:
            self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
        elif self.metric in ['cosine', 'dot_product']:
            self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

        self.index.hnsw.efConstruction = efConstruction

        self.doc_ids = []
        self.texts = []

    def set_embedder(self, embedder):
        """
        Set the embedding function after initialization.

        Args:
            embedder (callable): Function that converts list of texts to embeddings.
        """
        self.embedder = embedder

    def build(self, corpus, doc_ids):
        """
        Build the HNSW index from corpus texts.

        Args:
            corpus (List[str]): Documents.
            doc_ids (List[str]): Document IDs.
        """
        if self.embedder is None:
            raise ValueError("Embedder function is not set.")
        if len(corpus) != len(doc_ids):
            raise ValueError("Corpus and doc_ids must have the same length.")

        embeddings = self.embedder(corpus)
        embeddings = np.array(embeddings).astype('float32')

        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)

        self.index.reset()
        self.index.add(embeddings)

        self.doc_ids = doc_ids
        self.texts = corpus

    def save(self, filepath):
        """
        Save index and metadata to disk.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': faiss.serialize_index(self.index),
                'doc_ids': self.doc_ids,
                'texts': self.texts,
                'dim': self.dim,
                'metric': self.metric,
                'M': self.M,
                'efConstruction': self.efConstruction
            }, f)

    @classmethod
    def load(cls, filepath, embedder=None):
        """
        Load index and metadata from disk.

        Args:
            filepath (str): Path to saved index.
            embedder (callable): Embedder function (optional, can be set later).

        Returns:
            HNSWIndex instance.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        obj = cls(dim=data['dim'], embedder=embedder, metric=data['metric'],
                  M=data['M'], efConstruction=data['efConstruction'])
        obj.index = faiss.deserialize_index(data['index'])
        obj.doc_ids = data['doc_ids']
        obj.texts = data['texts']
        return obj

    def search(self, query_text, top_k=5):
        """
        Search index for nearest neighbors of query_text.

        Args:
            query_text (str): Query string.
            top_k (int): Number of results.

        Returns:
            List[Dict]: List of dicts with 'doc_id', 'text', and 'score'.
        """
        if self.embedder is None:
            raise ValueError("Embedder function is not set.")

        query_emb = self.embedder([query_text])
        query_emb = np.array(query_emb).astype('float32')

        if self.metric == 'cosine':
            faiss.normalize_L2(query_emb)

        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                'doc_id': self.doc_ids[idx],
                'text': self.texts[idx],
                'score': float(dist)
            })

        return results