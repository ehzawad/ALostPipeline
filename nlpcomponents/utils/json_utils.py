from __future__ import annotations

import json
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def safe_json_dump(data: Any, fp, **kwargs) -> None:
    kwargs.setdefault('default', json_default)
    json.dump(data, fp, **kwargs)


def safe_json_dumps(data: Any, **kwargs) -> str:
    kwargs.setdefault('default', json_default)
    return json.dumps(data, **kwargs)
