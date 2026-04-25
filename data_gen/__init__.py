"""
firewatch_agent.data_gen package.

Orchestrates dataset generation, human review, and HuggingFace upload.
Runs on developer's local machine only — never on Kaggle or Colab.
"""

__all__ = [
    "run_generator",
    "review",
    "upload",
    "validate",
]