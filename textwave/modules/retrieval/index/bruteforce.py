import numpy as np
import pickle
import faiss

class BruteForceIndex:
    """
    A brute-force FAISS index for storing and retrieving text document embeddings with doc_ids.

    Args:
        dim (int): Dimensionality of embeddings.
        embedder (callable, optional): Function converting list of texts to embeddings.
        metric (str): Distance metric, either 'euclidean' or 'cosine' (default='cosine').
    """

    def __init__(self, dim, embedder=None, metric='cosine'):
        self.dim = dim
        self.metric = metric.lower()
        self.embedder = embedder
        self.doc_ids = []
        self.texts = []

        if self.metric in ['euclidean', 'minkowski']:
            self.index = faiss.IndexFlatL2(dim)
        elif self.metric in ['cosine', 'dot_product']:
            self.index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

    def set_embedder(self, embedder):
        """
        Set or update the embedder function.

        Args:
            embedder (callable): Function converting list of texts to embeddings.
        """
        self.embedder = embedder

    def _check_embedder(self):
        if self.embedder is None:
            raise ValueError("Embedder function not set. Call `set_embedder()` first.")

    def build(self, corpus, doc_ids):
        """
        Build the FAISS brute-force index from a corpus and document IDs.

        Args:
            corpus (List[str]): List of document texts.
            doc_ids (List[str]): Corresponding list of document IDs.
        """
        self._check_embedder()

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
        Save the index and metadata to disk.

        Args:
            filepath (str): Path to save the index file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': faiss.serialize_index(self.index),
                'doc_ids': self.doc_ids,
                'texts': self.texts,
                'dim': self.dim,
                'metric': self.metric
            }, f)

    @classmethod
    def load(cls, filepath, embedder=None):
        """
        Load the index and metadata from disk.

        Args:
            filepath (str): Path to the saved index file.
            embedder (callable, optional): Embedder function for encoding queries.

        Returns:
            BruteForceIndex instance.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        obj = cls(dim=data['dim'], embedder=embedder, metric=data['metric'])
        obj.index = faiss.deserialize_index(data['index'])
        obj.doc_ids = data['doc_ids']
        obj.texts = data['texts']
        return obj

    def search(self, query_text, top_k=5):
        """
        Search for nearest neighbors of a query text.

        Args:
            query_text (str): Query string.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict]: List of dicts each containing 'doc_id', 'text', and 'score'.
        """
        self._check_embedder()

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
    
    def add(self, doc, doc_id):
        """
        Add a single document and its embedding to the index.

        Args:
            doc (str): Document text.
            doc_id (str): Corresponding document ID.
        """
        self._check_embedder()

        embedding = self.embedder([doc])
        embedding = np.array(embedding).astype('float32')

        if self.metric == 'cosine':
            faiss.normalize_L2(embedding)

        self.index.add(embedding)
        self.doc_ids.append(doc_id)
        self.texts.append(doc)