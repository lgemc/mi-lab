from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..core.config import ModelConfig, load_config

"""
What the explorer is currently pointed at, and the checkpoint it has not
loaded yet.

This is the equivalent of k9s's cluster connection: one object every view
reads its context from, so that switching model is one assignment rather than
a rebuild of the view stack.

The load is lazy on purpose and it is the decision this module exists to
encode. Runs and artifacts are readable with no torch and no checkpoint --
that is the whole point of the artifact format -- so the explorer must open
instantly on them and pay for a model only when a view actually needs one. An
explorer that loads a checkpoint at startup is one nobody opens to check what
ran last night.

A common pipe could be: Session | view.rows | drill down
"""

DEFAULT_PROMPT = "Then, Jack and Mary went to the store. Mary gave a drink to"

@dataclass
class Session:
    """The model, prompt and output root every view reads its context from"""
    model_id: str = "gpt2-small"
    prompt: str = DEFAULT_PROMPT
    root: str = "outputs"
    artifact: Any = None
    _config: Optional[ModelConfig] = None
    _adapter: Any = None
    _cache: Dict[str, Any] = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        """Whether a checkpoint is in memory, which is what the header reports"""
        return self._adapter is not None

    def config(self) -> ModelConfig:
        """The named config, resolved against configs/ but not against a checkpoint

        Cheap: this reads a YAML file. It is what the model list and the header
        show, and it is why `ie` can describe a model it has never loaded.
        """
        if self._config is None or self._config.id != self.model_id:
            self._config = load_config(self.model_id)
            self._adapter = None
            self._cache.clear()
        return self._config

    def adapter(self):
        """The loaded adapter, loading the checkpoint on first use

        Every caller of this is a view that cannot answer its question without
        the weights. Views that can answer without them must not call it, or
        the laziness above buys nothing.
        """
        if self._adapter is None:
            from ..model.adapter import load_adapter

            self._adapter = load_adapter(self.config())
            self._config = self._adapter.cfg
            self._cache.clear()
        return self._adapter

    def cached(self, key: str, produce):
        """Memoize one derived thing per session, keyed by name

        Capturing activations for the same prompt on every keystroke is what
        makes a terminal UI feel broken, and the cache is cleared whenever the
        model or the prompt changes so it can never answer for the wrong one.
        """
        if key not in self._cache:
            self._cache[key] = produce()
        return self._cache[key]

    def set_model(self, model_id: str) -> None:
        """Point the session at another config, dropping whatever was loaded"""
        self.model_id = model_id
        self._config = None
        self._adapter = None
        self._cache.clear()

    def set_prompt(self, prompt: str) -> None:
        """Change the text the token and activation views work over"""
        self.prompt = prompt
        self._cache.clear()

    def describe(self) -> str:
        """One line of context for the header, without loading anything"""
        try:
            cfg = self.config()
        except ValueError as error:
            return f"[!] {error}"
        sizes = f"{cfg.n_layers or '?'}L x {cfg.d_model or '?'}d x {cfg.n_heads or '?'}h"
        state = "loaded" if self.loaded else "not loaded"
        return f"{cfg.id} ({cfg.hf_name})  {sizes}  {cfg.device}/{cfg.dtype}  [{state}]"
