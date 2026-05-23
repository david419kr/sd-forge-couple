import math
from functools import wraps

import torch

from backend.nn.anima import SelfCrossAttention
from lib_couple.logging import logger
from modules.devices import device, dtype

from .attention_masks import get_dit_mask, lcm_for_list


def _squeeze_single_conditioning(tensor: torch.Tensor) -> torch.Tensor:
    while tensor.dim() > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def _pad_sequence_tensor(tensor: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tensor.shape[-2] >= target_tokens:
        return tensor

    return torch.nn.functional.pad(tensor, (0, 0, 0, target_tokens - tensor.shape[-2]))


def _cond_to_tensor(cond) -> torch.Tensor:
    if isinstance(cond, dict):
        for key in ("crossattn", "cross_attn"):
            tensor = cond.get(key)
            if torch.is_tensor(tensor):
                return tensor

    if torch.is_tensor(cond):
        return cond

    if isinstance(cond, (list, tuple)):
        if len(cond) == 1:
            return _cond_to_tensor(cond[0])

        tensors = [_squeeze_single_conditioning(_cond_to_tensor(item)) for item in cond]
        max_tokens = max(tensor.shape[-2] for tensor in tensors)
        tensors = [_pad_sequence_tensor(tensor, max_tokens) for tensor in tensors]
        return torch.stack(tensors, dim=0)

    raise TypeError(f"Unsupported Forge Couple Anima conditioning type: {type(cond)!r}")


def _fit_cond_batch(cond: torch.Tensor, batch_size: int) -> torch.Tensor:
    if cond.dim() == 2:
        cond = cond.unsqueeze(0)
    if cond.dim() == 4 and cond.shape[1] == 1:
        cond = cond.squeeze(1)

    if cond.shape[0] == batch_size:
        return cond

    if cond.shape[0] == 1:
        return cond.repeat(batch_size, 1, 1)

    repeats = (batch_size + cond.shape[0] - 1) // cond.shape[0]
    return cond.repeat(repeats, 1, 1)[:batch_size]


def _pad_to_token_multiple(cond: torch.Tensor, multiple: int = 512) -> torch.Tensor:
    target_tokens = math.ceil(cond.shape[-2] / multiple) * multiple
    return _pad_sequence_tensor(cond, target_tokens)


class AttentionCoupleAnima:

    @staticmethod
    @torch.inference_mode()
    def patch_dit(model, base_mask, width: int, height: int, kwargs: dict):
        dit = model.model.diffusion_model

        num_conds = len(kwargs) // 2 + 1

        mask = [base_mask] + [kwargs[f"mask_{i}"] for i in range(1, num_conds)]
        mask = torch.stack(mask, dim=0).to(device=device, dtype=dtype)

        if mask.sum(dim=0).min().item() <= 0.0:
            logger.error("Mask must be completely filled...")
            return None

        mask = mask / mask.sum(dim=0, keepdim=True)
        conds: list[torch.Tensor] = []

        for i in range(1, num_conds):
            c = _cond_to_tensor(kwargs[f"cond_{i}"])
            c = _fit_cond_batch(c, c.shape[0] if c.dim() > 2 else 1)
            c = _pad_to_token_multiple(c).to(device=device, dtype=dtype)
            conds.append(c)

        num_tokens = [cond.shape[-2] for cond in conds]

        SelfCrossAttention.couple_orig_forward = SelfCrossAttention.forward

        @wraps(SelfCrossAttention.couple_orig_forward)
        @torch.inference_mode()
        def couple_forward(
            self: "SelfCrossAttention",
            x: torch.Tensor,
            context: torch.Tensor,
            rope_emb: torch.Tensor,
            transformer_options: dict = {},
        ):
            if self.is_SelfAttn:
                return self.couple_orig_forward(
                    x=x,
                    context=context,
                    rope_emb=rope_emb,
                    transformer_options=transformer_options,
                )

            cond_or_unconds = transformer_options.get("cond_or_uncond", None)

            if context is None or not cond_or_unconds:
                return self.couple_orig_forward(
                    x=x,
                    context=context,
                    rope_emb=rope_emb,
                    transformer_options=transformer_options,
                )

            num_chunks = len(cond_or_unconds)
            batch_size = x.shape[0] // num_chunks

            x_chunks = x.chunk(num_chunks, dim=0)

            context_3d = context.squeeze(1)
            if (dim := math.ceil(context_3d.shape[1] / 512) * 512) > 512:
                context_3d = torch.nn.functional.pad(
                    context_3d,
                    (0, 0, 0, dim - context_3d.shape[1]),
                )

            ctx_seq_len = context_3d.shape[-2]
            context_chunks = context_3d.chunk(num_chunks, dim=0)

            lcm_tokens = lcm_for_list(num_tokens + [ctx_seq_len])
            assert lcm_tokens in (512, 1024), "Your prompt is way too long..."

            conds_tensor = torch.cat(
                [
                    _fit_cond_batch(cond, batch_size).repeat(
                        1, lcm_tokens // cond.shape[-2], 1
                    )
                    for cond in conds
                ],
                dim=0,
            )

            new_x = []
            new_context = []

            for idx, cond_or_uncond in enumerate(cond_or_unconds):
                c_target = context_chunks[idx].repeat(1, lcm_tokens // ctx_seq_len, 1)
                if cond_or_uncond == 1:
                    new_x.append(x_chunks[idx])
                    new_context.append(c_target)
                else:
                    new_x.append(x_chunks[idx].repeat(num_conds, 1, 1))
                    new_context.append(torch.cat([c_target, conds_tensor], dim=0))

            x_in = torch.cat(new_x, dim=0)
            ctx_in = torch.cat(new_context, dim=0).to(dtype=x_in.dtype)

            out = self.couple_orig_forward(
                x_in,
                context=ctx_in,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

            seq_len: int = out.shape[1]

            mask_downsample = get_dit_mask(
                mask, seq_len, width, height, patch_size=dit.patch_spatial
            )

            outputs = []
            pos = 0

            for idx, cond_or_uncond in enumerate(cond_or_unconds):
                if cond_or_uncond == 1:
                    outputs.append(out[pos : pos + batch_size])
                    pos += batch_size
                else:
                    chunk = out[pos : pos + num_conds * batch_size]
                    chunk = chunk.view(num_conds, batch_size, seq_len, -1)

                    masked_output = (chunk * mask_downsample).sum(dim=0)
                    outputs.append(masked_output)
                    pos += num_conds * batch_size

            return torch.cat(outputs, dim=0)

        couple_forward._couple = True
        SelfCrossAttention.forward = couple_forward

        return model

    @staticmethod
    def unpatch():
        if hasattr(SelfCrossAttention, "couple_orig_forward"):
            if getattr(SelfCrossAttention.forward, "_couple", False):
                SelfCrossAttention.forward = SelfCrossAttention.couple_orig_forward
            del SelfCrossAttention.couple_orig_forward
