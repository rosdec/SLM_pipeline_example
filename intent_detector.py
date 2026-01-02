"""
Intent detection using MiniLM embeddings.

Classifies employee reports into predefined HR categories.
"""

import ollama
import numpy as np


# Intent categories with descriptions
INTENTS = {
    "harassment": "Harassment, intimidation, or misconduct at work",
    "burnout": "Employee burnout, stress, or mental health risk",
    "policy_violation": "Internal policy or ethics violation",
    "performance": "Performance or role-related concern",
    "noise": "Irrelevant or non-HR related input"
}


class IntentDetector:
    """Detects intent from employee reports using embedding similarity."""
    
    def __init__(self, model: str = "all-minilm"):
        """
        Initialize the intent detector.
        
        Args:
            model: The embedding model to use (default: all-minilm)
        """
        self.model = model
        self.intents = INTENTS
        self.intent_keys = list(INTENTS.keys())
        self.intent_texts = list(INTENTS.values())
        
        print(f"[INIT] Loading HR intent embeddings ({model})...")
        self.intent_embeddings = [
            ollama.embeddings(model=self.model, prompt=text)["embedding"]
            for text in self.intent_texts
        ]
    
    @staticmethod
    def cosine_sim(a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def detect(self, text: str) -> str:
        """
        Detect the intent of an employee report.
        
        Args:
            text: The employee report text
            
        Returns:
            The detected intent category
        """
        emb = ollama.embeddings(
            model=self.model,
            prompt=text
        )["embedding"]
        
        scores = [self.cosine_sim(emb, e) for e in self.intent_embeddings]
        return self.intent_keys[int(np.argmax(scores))]
