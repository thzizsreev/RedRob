"""Re-export production embedder from pipeline (single source of truth)."""

from pipeline.instructor_onnx import InstructorONNX, load_embedder, unload_embedder

__all__ = ["InstructorONNX", "load_embedder", "unload_embedder"]
