from __future__ import annotations

import importlib
import pkgutil

_REGISTRY: dict[str, type] = {}


def register(key: str):
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Ingestor keys must be non-empty strings")

    def deco(cls):
        existing = _REGISTRY.get(key)
        if existing is not None and (
            existing.__module__,
            existing.__qualname__,
        ) != (cls.__module__, cls.__qualname__):
            raise ValueError(
                f"Ingestor key '{key}' already registered by "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        _REGISTRY[key] = cls
        return cls

    return deco


def get_ingestor_cls(key: str) -> type:
    if key not in _REGISTRY:
        raise KeyError(f"No ingestor registered as '{key}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def load_ingestor_cls(key: str) -> type:
    """Load the module named for an ingestor key, then return its registration.

    A new domain can therefore be added as ``domains/<key>.py`` without also
    changing a central module list.
    """
    if key not in _REGISTRY:
        from . import domains

        module_name = f"{domains.__name__}.{key}"
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            raise KeyError(
                f"No ingestor registered as '{key}'. Known: {sorted(_REGISTRY)}"
            ) from exc
    return get_ingestor_cls(key)


def discover_domains() -> None:
    """Eagerly import every module in ingestion.domains."""
    from . import domains

    for m in pkgutil.iter_modules(domains.__path__):
        importlib.import_module(f"{domains.__name__}.{m.name}")
