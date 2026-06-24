"""Re-export production embedder from tracks.instructor (single source of truth)."""

from tracks.instructor.core.onnx_embedder import InstructorONNX, load_embedder, unload_embedder

__all__ = ["InstructorONNX", "load_embedder", "unload_embedder"]
