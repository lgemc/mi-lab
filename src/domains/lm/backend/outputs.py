"""
What the model says: its tokens, its next-token logits, its continuations.

The three questions that need the tokenizer as much as the weights, and one
refusal. `single_token` is that refusal: the IOI design rests on a name being
one token, because a logit difference over a name split into several stops
measuring the thing it is named after, and failing loudly here is what keeps
that from being found later as a mysteriously flat result.

`chat_wrap` is driven by `chat` in the config rather than guessed from the
checkpoint name -- an instruct model's answer lives after its own generation
prompt, so sending it raw text is a different question and not a smaller one.

A common pipe could be: single_token | logits | logit_difference
"""

from typing import List, Optional, Sequence

import torch

from ....core.config import ConfigError
from .positions import _last_real


class OutputMixin:
    """Tokenizing, decoding, and the two ways of asking what comes next"""

    def chat_wrap(self, prompts: Sequence[str]) -> List[str]:
        """Put each prompt through the checkpoint's chat template, if it wants one

        An instruct model's answer lives after its own generation prompt. Sending
        raw text to one is not a smaller version of the same question, it is a
        different question -- so this is driven by `chat` in the config rather
        than guessed from the checkpoint name, and a base model is untouched.
        """
        if not getattr(self.cfg, "chat", False):
            return list(prompts)
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

    def generate(self, prompts: Sequence[str], max_new_tokens: Optional[int] = None, **kwargs) -> List[str]:
        """Greedily continue each prompt, returning only the new text"""
        input_ids, attention_mask = self._encode(self.chat_wrap(prompts), padding_side="left")
        completions = []
        for ids, mask in self._chunks(input_ids, attention_mask):
            with torch.no_grad():
                generated = self.model.generate(
                    ids,
                    attention_mask=mask,
                    max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs,
                )
            completions.extend(self.tokenizer.batch_decode(generated[:, ids.shape[1] :], skip_special_tokens=True))
        return completions

    def single_token(self, text: str) -> int:
        """The id of a string that is exactly one token, or an error naming what it split into

        The whole IOI design rests on this: names have to be one token each or
        the answer is spread over several logits and a logit difference stops
        measuring the thing it is named after. Failing loudly here is what
        keeps that from being discovered as a mysteriously flat result.
        """
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            pieces = self.tokenizer.convert_ids_to_tokens(ids)
            raise ConfigError(
                f"'{text}' is {len(ids)} tokens on '{self.cfg.id}' ({pieces}), not one; "
                "single-token names are what makes a logit difference readable"
            )
        return int(ids[0])

    def tokens(self, prompt: str) -> List[str]:
        """The prompt as the strings the model actually sees, for labelling an axis"""
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return [self.tokenizer.decode([token]) for token in ids]

    def outputs(self, inputs: Sequence[str]) -> torch.Tensor:
        """What this model produces for these inputs, which for a decoder is its next-token logits

        The CircuitAdapter name, delegating to the TokenAdapter one. A decoder's
        `outputs` *is* logits, so this is a name and not a second
        implementation -- the split exists so that `methods/` can ask what came
        out without claiming it is indexed by a vocabulary.
        """
        return self.logits(inputs)

    def logits(self, prompts: Sequence[str]) -> torch.Tensor:
        """Next-token logits at each prompt's final real token, as [batch, vocab]"""
        if not prompts:
            raise ConfigError("logits needs at least one prompt")
        input_ids, attention_mask = self._encode(prompts, padding_side="right")
        chunks = []
        for ids, mask in self._chunks(input_ids, attention_mask):
            with torch.no_grad():
                output = self.model(ids, attention_mask=mask, use_cache=False).logits
            chunks.append(_last_real(output, mask).float().cpu())
        return torch.cat(chunks, dim=0)
