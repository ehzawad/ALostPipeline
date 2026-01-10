from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, TypeVar, Generic
from loguru import logger

T = TypeVar('T')

@dataclass
class _PipelineSlot(Generic[T]):
    pipeline: T
    version: int = 0
    _ref_count: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _drain_condition: Optional[threading.Condition] = field(default=None, repr=False)

    def __post_init__(self):
        self._drain_condition = threading.Condition(self._lock)

    def acquire(self) -> None:
        with self._lock:
            self._ref_count += 1

    def release(self) -> int:
        with self._drain_condition:
            self._ref_count -= 1
            count = self._ref_count
            if count == 0:
                self._drain_condition.notify_all()
        return count

    @property
    def ref_count(self) -> int:
        with self._lock:
            return self._ref_count

    def wait_for_drain(self, timeout: Optional[float] = None) -> bool:
        with self._drain_condition:
            if self._ref_count == 0:
                return True

            return self._drain_condition.wait_for(
                lambda: self._ref_count == 0,
                timeout=timeout
            )

class PipelineHolder(Generic[T]):

    def __init__(self):
        self._lock = threading.RLock()
        self._current_slot: Optional[_PipelineSlot[T]] = None
        self._version_counter = 0

    def set(self, pipeline: T) -> None:
        with self._lock:
            self._version_counter += 1
            self._current_slot = _PipelineSlot(
                pipeline=pipeline,
                version=self._version_counter
            )
            logger.info(f"Pipeline set (version={self._version_counter})")

    def get(self) -> Optional[T]:
        with self._lock:
            if self._current_slot is None:
                return None
            return self._current_slot.pipeline

    @contextmanager
    def acquire(self):
        with self._lock:
            slot = self._current_slot
            if slot is not None:
                slot.acquire()

        try:
            yield slot.pipeline if slot else None
        finally:
            if slot is not None:
                slot.release()

    def swap(
        self,
        new_pipeline: T,
        drain_timeout: Optional[float] = 30.0,
        on_drain_timeout: str = "warn"
    ) -> Optional[T]:
        with self._lock:
            old_slot = self._current_slot
            self._version_counter += 1

            self._current_slot = _PipelineSlot(
                pipeline=new_pipeline,
                version=self._version_counter
            )
            new_version = self._version_counter
            old_version = old_slot.version if old_slot else None
            logger.info(f"Pipeline swapped (old={old_version}, new={new_version})")

        if old_slot is not None:
            refs = old_slot.ref_count
            if refs > 0:
                logger.info(f"Waiting for old pipeline (v{old_slot.version}) to drain ({refs} refs)...")
                drained = old_slot.wait_for_drain(timeout=drain_timeout)

                if not drained:
                    msg = f"Drain timeout after {drain_timeout}s (v{old_slot.version}, {old_slot.ref_count} refs)"
                    if on_drain_timeout == "raise":
                        raise TimeoutError(msg)
                    elif on_drain_timeout == "warn":
                        logger.warning(msg + " - returning old pipeline anyway")
                    else:
                        logger.warning(msg + " - forcing swap")
                else:
                    logger.info(f"Old pipeline (v{old_slot.version}) drained successfully")

            return old_slot.pipeline

        return None

    def clear(self, drain_timeout: Optional[float] = 30.0) -> Optional[T]:
        with self._lock:
            old_slot = self._current_slot
            self._current_slot = None
            logger.info("Pipeline cleared")

        if old_slot is not None:
            old_slot.wait_for_drain(timeout=drain_timeout)
            return old_slot.pipeline

        return None

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._current_slot is not None

    @property
    def current_version(self) -> int:
        with self._lock:
            if self._current_slot:
                return self._current_slot.version
            return 0

    @property
    def active_references(self) -> int:
        with self._lock:
            if self._current_slot:
                return self._current_slot.ref_count
            return 0
