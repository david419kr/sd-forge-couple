"""
Credit: laksjdjf
https://github.com/laksjdjf/cgem156-ComfyUI/blob/main/scripts/attention_couple/node.py

Modified by. Haoming02 to work with Forge
"""

import torch

from lib_couple.logging import logger
from modules.devices import device, dtype

from .attention_masks import get_mask, lcm_for_list


def _cond_to_tensor(cond) -> torch.Tensor:
    while isinstance(cond, (list, tuple)) and len(cond) == 1:
        cond = cond[0]

    if isinstance(cond, dict):
        for key in ("crossattn", "cross_attn"):
            tensor = cond.get(key)
            if torch.is_tensor(tensor):
                return tensor

    if torch.is_tensor(cond):
        return cond

    raise TypeError(f"Unsupported Forge Couple conditioning type: {type(cond)!r}")


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


class AttentionCouple:
    batch_size: int

    @classmethod
    @torch.inference_mode()
    def patch_unet(cls, model, base_mask, kwargs: dict):
        m = model.clone()

        num_conds = len(kwargs) // 2 + 1

        mask = [base_mask] + [kwargs[f"mask_{i}"] for i in range(1, num_conds)]
        mask = torch.stack(mask, dim=0).to(device=device, dtype=dtype)

        if mask.sum(dim=0).min().item() <= 0.0:
            logger.error("Mask must be completely filled...")
            return None

        mask = mask / mask.sum(dim=0, keepdim=True)

        conds = [
            _cond_to_tensor(kwargs[f"cond_{i}"]).to(device=device, dtype=dtype)
            for i in range(1, num_conds)
        ]
        num_tokens = [cond.shape[-2] for cond in conds]

        @torch.inference_mode()
        def attn2_patch(q, k, v, extra_options):
            assert torch.allclose(k, v), "k and v should be the same"

            cond_or_unconds = extra_options["cond_or_uncond"]
            num_chunks = len(cond_or_unconds)
            cls.batch_size = q.shape[0] // num_chunks
            q_chunks = q.chunk(num_chunks, dim=0)
            k_chunks = k.chunk(num_chunks, dim=0)
            lcm_tokens = lcm_for_list(num_tokens + [k.shape[1]])
            conds_tensor = torch.cat(
                [
                    _fit_cond_batch(cond, cls.batch_size).repeat(
                        1, lcm_tokens // num_tokens[i], 1
                    )
                    for i, cond in enumerate(conds)
                ],
                dim=0,
            )

            qs, ks = [], []
            for i, cond_or_uncond in enumerate(cond_or_unconds):
                k_target = k_chunks[i].repeat(1, lcm_tokens // k.shape[1], 1)
                if cond_or_uncond == 1:  # uncond
                    qs.append(q_chunks[i])
                    ks.append(k_target)
                else:
                    qs.append(q_chunks[i].repeat(num_conds, 1, 1))
                    ks.append(torch.cat([k_target, conds_tensor], dim=0))

            qs = torch.cat(qs, dim=0).to(q)
            ks = torch.cat(ks, dim=0).to(k)

            if qs.size(0) % 2 == 1:
                empty = torch.zeros_like(qs[0]).unsqueeze(0)
                qs = torch.cat((qs, empty), dim=0)
                empty = torch.zeros_like(ks[0]).unsqueeze(0)
                ks = torch.cat((ks, empty), dim=0)

            return qs, ks, ks

        @torch.inference_mode()
        def attn2_output_patch(out, extra_options):
            cond_or_unconds = extra_options["cond_or_uncond"]
            mask_downsample = get_mask(
                mask, cls.batch_size, out.shape[1], extra_options["original_shape"]
            )
            outputs = []
            pos = 0
            for cond_or_uncond in cond_or_unconds:
                if cond_or_uncond == 1:  # uncond
                    outputs.append(out[pos : pos + cls.batch_size])
                    pos += cls.batch_size
                else:
                    masked_output = (
                        out[pos : pos + num_conds * cls.batch_size] * mask_downsample
                    ).view(num_conds, cls.batch_size, out.shape[1], out.shape[2])
                    masked_output = masked_output.sum(dim=0)
                    outputs.append(masked_output)
                    pos += num_conds * cls.batch_size
            return torch.cat(outputs, dim=0)

        m.set_model_attn2_patch(attn2_patch)
        m.set_model_attn2_output_patch(attn2_output_patch)

        return m
