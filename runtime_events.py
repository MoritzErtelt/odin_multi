"""Optional process-local timing events; no framework or platform dependencies."""

import json
import os
import time
from pathlib import Path


def emit(event: str, **fields) -> None:
    destination = os.environ.get("ODIN_TIMING_LOG")
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "event": event,
        "timestamp": time.time(),
        "monotonic": time.perf_counter(),
        "pid": os.getpid(),
        **fields,
    }
    with path.open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
