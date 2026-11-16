"""Small registry for independently testable possession modules."""


class ModuleRegistry:
    """Map stable module names to factories without changing submission flow."""

    def __init__(self):
        self._factories = {}

    def register(self, name, factory, *, replace=False):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("module name must be a nonempty string")
        if not callable(factory):
            raise TypeError("module factory must be callable")
        if name in self._factories and not replace:
            raise ValueError(f"module {name!r} is already registered")
        self._factories[name] = factory

    def create(self, name, **kwargs):
        try:
            factory = self._factories[name]
        except KeyError as error:
            raise KeyError(f"unknown possession module {name!r}") from error
        return factory(**kwargs)

    def names(self):
        return tuple(sorted(self._factories))


def create_default_registry():
    """Build a registry containing the advanced adapter and optional baseline."""
    # Lazy imports keep the registry usable without introducing module cycles.
    from modules.nearest_player import NearestPlayer
    from modules.synx_combined import SynxCombined

    registry = ModuleRegistry()
    registry.register("synx_combined", SynxCombined)
    registry.register("nearest_player", NearestPlayer)
    return registry


DEFAULT_REGISTRY = create_default_registry()
