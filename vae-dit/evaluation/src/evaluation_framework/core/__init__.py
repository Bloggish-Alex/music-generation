"""Shared artifact-only primitives used by evaluation modules."""

from .artifacts import ArtifactReference, VerifiedArtifactResolver
from .tensor_schema import SemanticTensorDecoder

__all__ = ["ArtifactReference", "SemanticTensorDecoder", "VerifiedArtifactResolver"]
