"""Multi-layer CSA injection via forward hooks.

CSA is applied at multiple transformer layer boundaries during the forward
pass. A single shared CSA module fires at every selected layer, with a
per-layer learnable scalar gate (initialized to 0) that scales how much
cross-batch information is injected back into the residual stream.

Why hooks instead of subclassing the model:
- Works with any HF causal LM without monkey-patching architecture
- Easy to remove/disable for ablation
- DDP-friendly: shared CSA params get one set of gradients per step

Why a single shared CSA across layers:
- Keeps trainable param count bounded (~60M instead of layers*60M)
- Forces the same routing mechanism at every depth, which simplifies
  what the module has to learn
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the list of transformer decoder layers on a HF model.

    Tries common attribute paths for Llama / Qwen / GPT-style architectures.
    """
    base = None
    for attr in ("model", "transformer", "decoder", "base_model"):
        candidate = getattr(model, attr, None)
        if candidate is not None:
            base = candidate
            break
    if base is None:
        base = model

    for attr in ("layers", "h", "block", "decoder_layers"):
        layers = getattr(base, attr, None)
        if layers is not None and hasattr(layers, "__len__"):
            return layers

    raise ValueError(
        "Could not find decoder layer list on model. Tried .model.layers, "
        ".transformer.h, .decoder.layers, .base_model.layers."
    )


def default_layer_indices(num_layers: int, every: int = 4) -> List[int]:
    """Insertion points: last layer of each `every`-block.

    For 28 layers with every=4, returns [3, 7, 11, 15, 19, 23, 27].
    """
    return [i for i in range(num_layers) if (i + 1) % every == 0]


class MultiLayerCSAModule(nn.Module):
    """Wraps a single CSA submodule + per-layer alpha gates + hook lifecycle.

    Owns the hook registration/removal and the per-layer scalar gates. The
    underlying `csa` submodule (e.g. CrossSequenceAttentionV2) is shared
    across all selected layers — only one set of Q/K/V/out weights.

    Per-layer gates are initialized to 0, so before any training the model
    behaves identically to the base model (cross-batch contribution = 0).
    """

    def __init__(
        self,
        csa: nn.Module,
        layer_indices: List[int],
    ):
        super().__init__()
        self.csa = csa
        self.layer_indices = list(layer_indices)
        self.layer_alphas = nn.Parameter(torch.zeros(len(self.layer_indices)))
        self._idx_to_alpha = {idx: i for i, idx in enumerate(self.layer_indices)}

        # Per-forward state. Set via set_context(...) before model forward,
        # cleared via clear_context(). Not persisted across steps.
        self._question_emb: Optional[torch.Tensor] = None
        self._attention_mask: Optional[torch.Tensor] = None
        # Master switch. When False the hook returns input unchanged
        # regardless of batch size — used to evaluate "Independent" baseline.
        self._enabled: bool = True

        self._handles: list = []

    def set_context(
        self,
        question_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Set per-forward inputs. Call before model.forward.

        - question_emb [B, d]: routing embedding for CSA. If None, uses
          mean-pooled current hidden as routing query.
        - attention_mask [B, L]: used to mean-pool only valid positions when
          building the per-chunk summary inside the hook. If None, mean over
          all positions.
        """
        self._question_emb = question_emb
        self._attention_mask = attention_mask

    def clear_context(self):
        self._question_emb = None
        self._attention_mask = None

    def set_enabled(self, enabled: bool):
        """Master on/off for the cross-batch injection."""
        self._enabled = bool(enabled)

    def _summary_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Mean-pool per chunk over the sequence dimension. Returns [B, d]."""
        mask = self._attention_mask
        if mask is not None and mask.dim() == 2 and mask.shape[1] == hidden.shape[1]:
            m = mask.to(hidden.dtype).unsqueeze(-1)
            return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return hidden.mean(dim=1)

    def _make_hook(self, layer_idx: int):
        alpha_idx = self._idx_to_alpha[layer_idx]

        def hook(module, args, output):
            # Output may be tensor or tuple (most HF decoder layers return tuple)
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            if not torch.is_tensor(hidden) or hidden.dim() != 3:
                return output

            if not self._enabled:
                return output

            B = hidden.size(0)

            # B < 2: no cross-batch info available. Still pass CSA params
            # through computation graph if training so DDP doesn't deadlock.
            if B < 2:
                if self.training:
                    summary = self._summary_from_hidden(hidden)
                    q_emb = (
                        self._question_emb
                        if self._question_emb is not None
                        else summary
                    )
                    _ = self.csa(summary, question_emb=q_emb).sum() * 0.0
                    _ = self.layer_alphas[alpha_idx] * 0.0
                return output

            summary = self._summary_from_hidden(hidden)  # [B, d]
            q_emb = (
                self._question_emb
                if self._question_emb is not None
                else summary
            )
            mixed = self.csa(summary, question_emb=q_emb)  # [B, d]
            delta = (mixed - summary).unsqueeze(1)  # [B, 1, d]

            alpha = self.layer_alphas[alpha_idx]
            new_hidden = hidden + alpha * delta

            if rest is None:
                return new_hidden
            return (new_hidden,) + rest

        return hook

    def register(self, model: nn.Module):
        """Attach hooks to the selected decoder layers of `model`."""
        if self._handles:
            raise RuntimeError("Hooks already registered. Call unregister() first.")
        layers = _get_decoder_layers(model)
        for i in range(len(layers)):
            if i in self._idx_to_alpha:
                handle = layers[i].register_forward_hook(self._make_hook(i))
                self._handles.append(handle)
        logger.info(
            "Registered CSA hooks at %d / %d layers (indices=%s)",
            len(self._handles), len(layers), self.layer_indices,
        )

    def unregister(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def alphas_summary(self) -> str:
        """For logging: comma-separated current alpha values."""
        return ", ".join(f"{a.item():+.4f}" for a in self.layer_alphas)
