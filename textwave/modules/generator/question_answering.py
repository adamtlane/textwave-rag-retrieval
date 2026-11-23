import os
import time
import numpy as np
from typing import List, Tuple, Union
from mistralai import Mistral

class QAGeneratorMistral:
    """
    A question-answer generator that uses the Mistral API to generate answers
    based on provided context and a user-specified query.
    """

    def __init__(self, api_key: str, temperature: float = 0.3, generator_model: str = "mistral-small-latest"):
        """
        Initializes the QAGeneratorMistral class with credentials and settings.

        Parameters:
            api_key (str): API key string for authenticating with the Mistral API.
            temperature (float): Controls randomness in the model's output; lower is more deterministic.
            generator_model (str): Identifier for which Mistral model to use.
        """
        self.api_key = api_key
        self.temperature = temperature
        self.generator_model = generator_model
        
        self.client = Mistral(api_key=api_key) # Instantiate a client for interacting with the Mistral API.

    def generate_answer(self, query: str, context: List[str]) -> str:
        """
        Generates an answer using Mistral's chat API based solely on the provided context.

        Parameters:
            query (str): The question to answer.
            context (List[str]): A list of context strings that the model must base its answer on.

        Returns:
            str: A generated answer string based only on the given context.
        """
        
        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context))
        combined_input = (
            (f"Context:\n{context_block}\n\n" if context else "")
            + f"Question: {query}\nAnswer:"
        ) # Combine the retrieved context (if any) and the query into a single prompt string.

        
        style = ("Reply with the shortest answer possible, a few words copied from the context when you can. "
                 "For yes or no questions, reply with only yes or no. "
                 "Do not write a full sentence or explain your answer.")
        if context: # Retrieval augmented: answer only from the retrieved passages.
            system_prompt = ("Answer the question using only the provided context. " + style +
                             " If the context does not contain the answer, reply with: unknown")
        else: # Closed book: no passages were retrieved, so answer from general knowledge.
            system_prompt = "Answer the question from your own knowledge. " + style.replace(" copied from the context when you can", "")

        chat_response = self.client.chat.complete(
            model=self.generator_model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": combined_input,
                },
            ]
        ) # Submit the prompt to Mistral's chat API with a system instruction.

        time.sleep(2) # Optional throttling to avoid exceeding API rate limits.

        return chat_response.choices[0].message.content # Return the content of the first response message.


if __name__ == "__main__": # Set up Mistral-based generator and ask the question
    
    API_KEY = os.environ["MISTRAL_API_KEY"] # Read the Mistral API key from an environment variable.
    generator_mistral = QAGeneratorMistral(api_key=API_KEY) # Instantiate the Mistral-based QA generator.

    context = [
        "Albert Einstein was a theoretical physicist born in Germany.",
        "He developed the theory of relativity, one of the two pillars of modern physics.",
        "Einstein was awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect."
    ] # Provide context passages relevant to the query.

    query = "What was Einstein awarded the Nobel Prize for?" # Define the user query to be answered using only the context.

    answer = generator_mistral.generate_answer(query=query, context=context) # Use the generator to produce an answer.
    
    print("=== Mistral generator ===")
    print(f"Question: {query}")
    print(f"Answer: {answer}\n") # Display the question and generated answer.