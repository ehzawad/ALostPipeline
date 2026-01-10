from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, FrozenSet, IO, Tuple, Union

class RestrictedUnpickler(pickle.Unpickler):

    ALLOWED_NUMPY: FrozenSet[Tuple[str, str]] = frozenset({
        ('numpy.core.multiarray', '_reconstruct'),
        ('numpy._core.multiarray', '_reconstruct'),
        ('numpy._core.multiarray', 'scalar'),
        ('numpy._core.numeric', '_frombuffer'),
        ('numpy', 'ndarray'),
        ('numpy', 'dtype'),
        ('numpy', 'float64'),
        ('numpy', 'float32'),
        ('numpy', 'int64'),
        ('numpy', 'int32'),
        ('numpy', 'bool_'),
    })

    ALLOWED_BUILTINS: FrozenSet[str] = frozenset({
        'dict', 'list', 'tuple', 'set', 'frozenset',
    })

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self.ALLOWED_NUMPY:
            mod = __import__(module, fromlist=[name])
            return getattr(mod, name)

        if module == 'builtins' and name in self.ALLOWED_BUILTINS:
            return getattr(__import__(module), name)

        raise pickle.UnpicklingError(
            f"Unsafe pickle: module={module}, name={name}. "
            "Model file may be corrupted or malicious. "
            "Only numpy arrays and basic Python types are allowed."
        )

def safe_load(file: Union[str, Path, IO[bytes]]) -> Any:
    if isinstance(file, (str, Path)):
        with open(file, 'rb') as f:
            return RestrictedUnpickler(f).load()
    else:
        return RestrictedUnpickler(file).load()
