import re
import numpy as np
from collections import Counter
from typing import List


class BagOfWords:
    """
    A Bag-of-Words representation transformer.

    This class learns a vocabulary from a corpus of documents and can transform
    new documents into their Bag-of-Words vector representations using that vocabulary.
    """

    def __init__(self):
        """
        Initializes the BagOfWords instance.

        Attributes:
            vocabulary_ (dict or None): A mapping from tokens (str) to indices (int),
                                        initialized during fitting.
        """
        self.vocabulary_ = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes input text into lowercase alphanumeric words.

        Parameters:
            text (str): The input string to tokenize.

        Returns:
            List[str]: A list of lowercase tokens (words) extracted from the text.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, documents: List[str]) -> "BagOfWords":
        """
        Learns a vocabulary dictionary from the list of documents.

        Parameters:
            documents (List[str]): A list of text documents to learn the vocabulary from.

        Returns:
            BagOfWords: The instance itself, now with the vocabulary_ attribute set.
        """
        unique_tokens = set()

        for doc in documents:
            tokens = self._tokenize(doc) # Collect all unique tokens from all documents.
            unique_tokens.update(tokens)

        sorted_tokens = sorted(unique_tokens) # Sort tokens alphabetically to assign consistent indices.
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted_tokens)} # Map each token to a unique index.

        return self

    def transform(self, document: str) -> np.ndarray:
        """
        Transforms a single document into its Bag-of-Words vector.

        Parameters:
            document (str): The text document to be transformed.

        Returns:
            np.ndarray: A 1D NumPy array representing the Bag-of-Words count vector.
        """
        if not self.vocabulary_:
            raise AttributeError("BagOfWords instance is not fitted yet. Call 'fit' first.")

        tokens = self._tokenize(document) # Tokenize input document.
        token_counts = Counter(tokens) # Count frequency of each token.

        vector = np.zeros(len(self.vocabulary_), dtype=int) # Initialize zero vector for the full vocabulary size.
        
        for token, count in token_counts.items(): # Populate vector with token counts.
            if token in self.vocabulary_:
                index = self.vocabulary_[token]
                vector[index] = count

        return vector

if __name__ == "__main__":

    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Never jump over the lazy dog quickly.",
        "A quick movement of the enemy will jeopardize six gunboats.",
        "All that glitters is not gold.",
        "To be or not to be, that is the question.",
        "I think, therefore I am.",
        "The only thing we have to fear is fear itself.",
        "Ask not what your country can do for you; ask what you can do for your country.",
        "That's one small step for man, one giant leap for mankind.",
    ] # Sample corpus of various famous sentences.

    transform = BagOfWords() # Initialize and fit the BoW transformer.
    transform.fit(corpus)

    test_document = "The quick dog jumps high over the lazy fox." # Test document to transform using the fitted vocabulary.
    bow_test = transform.transform(test_document)

    print("BoW vector:\n", bow_test) # Output the resulting Bag-of-Words vector.