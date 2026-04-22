import torch
from typing import Dict, List, Optional, Callable


class HookManager:
    """Manages PyTorch forward hooks for capturing activations."""

    def __init__(self):
        # list of PyTorch hook handles, stored so we can remove them later
        self._hooks = []
        # dictionary of layer outputs captured during the forward pass, keyed by layer name
        self._activations: Dict[str, torch.Tensor] = {}
        # user-defined hook functions for custom behavior beyond just
        self._custom_hooks: Dict[str, Callable] = {}

    @property
    def activations(self) -> Dict[str, torch.Tensor]:
        return self._activations

    def attach(self, model: torch.nn.Module, layer_names: List[str]) -> None:
        """Attach forward hooks to specified layers by name."""
        self.clear()
        available = dict(model.named_modules())

        for name in layer_names:
            if name not in available:
                raise ValueError(
                    f"Layer '{name}' not found. "
                    f"Available layers: {list(available.keys())}"
                )
            hook = available[name].register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def attach_all(self, model: torch.nn.Module) -> None:
        """Attach hooks to all layers in the model."""
        layer_names = [name for name, _ in model.named_modules() if name]
        self.attach(model, layer_names)

    def attach_by_type(self, model: torch.nn.Module, layer_type: type) -> None:
        """Attach hooks to all layers of a specific type (e.g., nn.Linear)."""
        layer_names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, layer_type)
        ]
        self.attach(model, layer_names)

    def attach_custom(
        self, model: torch.nn.Module, layer_name: str, hook_fn: Callable
    ) -> None:
        """Attach a custom hook function to a specific layer."""
        available = dict(model.named_modules())
        if layer_name not in available:
            raise ValueError(f"Layer '{layer_name}' not found.")

        hook = available[layer_name].register_forward_hook(hook_fn)
        self._hooks.append(hook)
        self._custom_hooks[layer_name] = hook_fn

    def get(self, layer_name: str) -> Optional[torch.Tensor]:
        """Retrieve captured activation for a specific layer."""
        return self._activations.get(layer_name)

    def get_shapes(self) -> Dict[str, torch.Size]:
        """Return the shape of each captured activation."""
        return {name: tensor.shape for name, tensor in self._activations.items()}

    def reset_activations(self) -> None:
        """Clear captured activations without removing hooks."""
        self._activations.clear()

    def clear(self) -> None:
        """Remove all hooks and clear activations."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations.clear()
        self._custom_hooks.clear()

    def _make_hook(self, name: str) -> Callable:
        """Create a hook function that stores the output activation."""

        def hook_fn(module, input, output):
            # HuggingFace models can return dataclass-like objects
            if hasattr(output, "last_hidden_state"):
                self._activations[name] = output.last_hidden_state.detach()
            elif isinstance(output, tuple):
                self._activations[name] = output[0].detach()
            elif isinstance(output, torch.Tensor):
                self._activations[name] = output.detach()

        return hook_fn

    def __len__(self) -> int:
        return len(self._hooks)

    def __del__(self):
        self.clear()
