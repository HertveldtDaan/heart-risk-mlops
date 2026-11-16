"""Shared code for the heart-risk MLOps project.

This package is used by the Azure ML components (prep, train, evaluate)
AND copied into the FastAPI image, so training and serving use exactly the
same feature contract and cleaning rules (no training-serving skew).
"""
