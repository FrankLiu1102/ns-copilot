"""
Progress event system for NS-Copilot pipeline.

Captures intermediate pipeline outputs (Planner decisions, Codex results,
generated code, early stopping, etc.) for display in the web UI.

Thread-safe: events can be emitted from a background thread and polled
from the main Gradio thread for real-time streaming.
"""

import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ProgressEvent:
    """A single progress event from the pipeline."""
    stage: str       # e.g., "planner", "codex", "code_generator", "execution", "early_stop", "controller", "analysis"
    title: str       # Display title
    content: str     # Markdown-formatted content
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """Thread-safe collector for progress events during pipeline execution."""

    def __init__(self):
        self._events: List[ProgressEvent] = []
        self._lock = threading.Lock()

    def emit(self, stage: str, title: str, content: str, **metadata):
        with self._lock:
            self._events.append(ProgressEvent(
                stage=stage, title=title, content=content, metadata=metadata
            ))

    def clear(self):
        with self._lock:
            self._events.clear()

    def get_events(self) -> List[ProgressEvent]:
        with self._lock:
            return list(self._events)

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)


# Global singleton for the current pipeline run
_current_tracker: Optional[ProgressTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> ProgressTracker:
    global _current_tracker
    with _tracker_lock:
        if _current_tracker is None:
            _current_tracker = ProgressTracker()
        return _current_tracker


def reset_tracker():
    global _current_tracker
    with _tracker_lock:
        _current_tracker = ProgressTracker()
