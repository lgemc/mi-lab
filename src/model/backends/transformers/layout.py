"""
Where the sites are, in whatever this architecture calls them.

This is the whole of the backend's architecture knowledge, and it is one file
so that "teaching the backend a new model family" is a diff nobody has to read
the rest of the package to make. Every lookup tries the names the major
families use and raises with the module's own type when none matches, so a new
checkpoint fails by naming what it is rather than by an AttributeError several
frames deeper.

Nothing above this layer learns that a model was a GPT-2 rather than a Pythia.

A common pipe could be: _blocks | _attention_projection | register_forward_hook
"""

from typing import Sequence

import torch

from ....core.config import ConfigError


def _first_attribute(root, paths: Sequence[str]):
    """Follow the first dotted attribute path that resolves, or return None"""
    for path in paths:
        module = root
        for attribute in path.split("."):
            module = getattr(module, attribute, None)
            if module is None:
                break
        if module is not None:
            return module
    return None

def _blocks(model) -> torch.nn.ModuleList:
    """Find the list of decoder blocks, whatever this architecture calls it

    Every residual stream hook site in this backend is a block boundary, so
    this is the only piece of architecture knowledge probing needs.
    """
    found = _first_attribute(model, ("transformer.h", "model.layers", "gpt_neox.layers", "model.decoder.layers"))
    if found is None:
        raise ConfigError(f"cannot find the decoder blocks of {type(model).__name__}; teach _blocks its layout")
    return found

def _attention_projection(block, index: int):
    """The linear that mixes the heads back into the residual stream

    This is the hinge every head-level operation turns on. Its *input* is the
    heads' outputs laid end to end -- n_heads contiguous slices of d_head --
    so reading it splits the heads apart and writing it patches one head
    without reimplementing attention. Its *output* is the whole attention
    write, so nothing downstream has to be told a patch happened.
    """
    found = _first_attribute(
        block, ("attn.c_proj", "self_attn.o_proj", "attention.dense", "attn.out_proj", "self_attention.dense")
    )
    if found is None:
        raise ConfigError(
            f"cannot find the attention output projection of block {index} "
            f"({type(block).__name__}); teach _attention_projection its layout"
        )
    return found

def _mlp(block, index: int):
    """The submodule whose output is this block's MLP write into the residual stream"""
    found = _first_attribute(block, ("mlp", "feed_forward", "ffn"))
    if found is None:
        raise ConfigError(f"cannot find the MLP of block {index} ({type(block).__name__}); teach _mlp its layout")
    return found

def _attention_norm(block, index: int):
    """The norm a block applies before attention reads the residual stream

    Its *input* is the residual as attention sees it, which is the destination
    end of every edge into this layer's attention. Nodes need only the
    projection above; an edge needs to know what the destination read, because
    ablating an edge changes one reader's input and leaves every other reader's
    alone -- which is the whole difference between an edge and a node.
    """
    found = _first_attribute(block, ("ln_1", "input_layernorm", "ln1", "norm1", "input_layer_norm"))
    if found is None:
        raise ConfigError(
            f"cannot find the pre-attention norm of block {index} ({type(block).__name__}); "
            "teach _attention_norm its layout"
        )
    return found

def _mlp_norm(block, index: int):
    """The norm a block applies before its MLP reads the residual stream

    The second destination in a block, and the one that sees this layer's own
    attention write. A source in layer L reaches L's MLP and not L's attention.
    """
    found = _first_attribute(
        block, ("ln_2", "post_attention_layernorm", "ln2", "norm2", "post_attention_layer_norm")
    )
    if found is None:
        raise ConfigError(
            f"cannot find the pre-MLP norm of block {index} ({type(block).__name__}); "
            "teach _mlp_norm its layout"
        )
    return found

def _final_norm(model):
    """The normalization sitting between the last block and the unembedding"""
    found = _first_attribute(
        model,
        (
            "transformer.ln_f", "model.norm", "gpt_neox.final_layer_norm",
            "model.decoder.final_layer_norm", "model.final_layernorm", "transformer.norm_f",
        ),
    )
    if found is None:
        raise ConfigError(f"cannot find the final norm of {type(model).__name__}; teach _final_norm its layout")
    return found
