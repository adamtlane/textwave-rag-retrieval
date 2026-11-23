import numpy as np
import pickle
import faiss

class LSHIndex:
    """
    FAISS LSH index wrapper for approximate nearest neighbor search on text embeddings.

    Args:
        dim (int): Embedding dimensionality.
        embedder (callable): Function that converts list of texts to embeddings.
        nbits (int): Number of bits for hashing (default=128).
    """

    def __init__(self, dim, embedder=None, nbits=128):
        self.dim = dim
        self.embedder = embedder
        self.nbits = nbits

        self.index = faiss.IndexLSH(dim, nbits)

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
        Build LSH index from corpus.

        Args:
            corpus (List[str]): Document texts.
            doc_ids (List[str]): Document IDs.
        """
        if self.embedder is None:
            raise ValueError("Embedder function is not set.")
        if len(corpus) != len(doc_ids):
            raise ValueError("Corpus and doc_ids must have the same length.")

        embeddings = self.embedder(corpus)
        embeddings = np.array(embeddings).astype('float32')

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
                'nbits': self.nbits
            }, f)

    @classmethod
    def load(cls, filepath, embedder=None):
        """
        Load index and metadata from disk.

        Args:
            filepath (str): File path.
            embedder (callable): Embedder function (optional).

        Returns:
            LSHIndex instance.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        obj = cls(dim=data['dim'], embedder=embedder, nbits=data['nbits'])
        obj.index = faiss.deserialize_index(data['index'])
        obj.doc_ids = data['doc_ids']
        obj.texts = data['texts']
        return obj

    def search(self, query_text, top_k=5):
        """
        Search index for nearest neighbors to query_text.

        Args:
            query_text (str): Query string.
            top_k (int): Number of results.

        Returns:
            List[Dict]: Each dict has 'doc_id', 'text', 'score'.
        """
        if self.embedder is None:
            raise ValueError("Embedder function is not set.")

        query_emb = self.embedder([query_text])
        query_emb = np.array(query_emb).astype('float32')

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