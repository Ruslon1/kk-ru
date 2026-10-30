"""Export a P0 checkpoint to ONNX encoder and decoder graphs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn

from kk_ru.config import load_config
from kk_ru.model import build_model
from kk_ru.tokenizer import load_tokenizer, validate_vocab


class ExportRMSNorm(nn.Module):
    def __init__(self, source: nn.RMSNorm):
        super().__init__()
        self.weight = source.weight
        self.eps = source.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized * self.weight


def replace_rms_norms(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, ExportRMSNorm(child))
        else:
            replace_rms_norms(child)


def export_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    if enable_gqa:
        repeats = q.shape[-3] // k.shape[-3]
        k = k.repeat_interleave(repeats, dim=-3)
        v = v.repeat_interleave(repeats, dim=-3)
    factor = scale if scale is not None else q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * factor
    if is_causal:
        q_len, k_len = q.shape[-2], k.shape[-2]
        causal = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril(
            diagonal=k_len - q_len
        )
        attn_mask = causal if attn_mask is None else attn_mask & causal
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + attn_mask
    return torch.matmul(torch.softmax(scores, dim=-1), v)


class EncoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.model.encode(src_ids, src_mask)


class DecoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.decode(tgt_ids, memory, src_mask, tgt_mask=tgt_mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export P0 as ONNX encoder/decoder graphs")
    parser.add_argument("--config", default="configs/p0.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/p0/best")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def export_graphs(cfg, model, output_dir: Path, opset: int) -> None:
    torch.manual_seed(0)
    batch, src_len, tgt_len = 1, 5, 4
    src_ids = torch.randint(0, cfg.model.vocab, (batch, src_len), dtype=torch.long)
    src_mask = torch.ones(batch, src_len, dtype=torch.bool)
    tgt_ids = torch.randint(0, cfg.model.vocab, (batch, tgt_len), dtype=torch.long)
    tgt_mask = torch.ones(batch, tgt_len, dtype=torch.bool)
    memory = model.encode(src_ids, src_mask)

    encoder_path = output_dir / "encoder.onnx"
    decoder_path = output_dir / "decoder.onnx"
    common = {"opset_version": opset, "dynamo": False}
    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        temp_dir = Path(temp_dir)
        temp_encoder = temp_dir / "encoder.onnx"
        temp_decoder = temp_dir / "decoder.onnx"
        encoder_model = EncoderGraph(copy.deepcopy(model))
        encoder_model.model.embed_tokens.weight = nn.Parameter(
            model.embed_tokens.weight.detach().clone(), requires_grad=False
        )
        torch.onnx.export(
            encoder_model,
            (src_ids, src_mask),
            temp_encoder,
            input_names=["src_ids", "src_mask"],
            output_names=["memory"],
            dynamic_axes={
                "src_ids": {0: "batch", 1: "source_length"},
                "src_mask": {0: "batch", 1: "source_length"},
                "memory": {0: "batch", 1: "source_length"},
            },
            **common,
        )
        decoder_model = DecoderGraph(copy.deepcopy(model))
        decoder_model.model.embed_tokens.weight = nn.Parameter(
            model.embed_tokens.weight.detach().clone(), requires_grad=False
        )
        if decoder_model.model.output is None:
            decoder_model.model.output = nn.Linear(
                cfg.model.d_model, cfg.model.vocab, bias=False
            )
            decoder_model.model.output.weight = nn.Parameter(
                model.embed_tokens.weight.detach().clone(), requires_grad=False
            )
        torch.onnx.export(
            decoder_model,
            (tgt_ids, memory, src_mask, tgt_mask),
            temp_decoder,
            input_names=["tgt_ids", "memory", "src_mask", "tgt_mask"],
            output_names=["logits"],
            dynamic_axes={
                "tgt_ids": {0: "batch", 1: "target_length"},
                "memory": {0: "batch", 1: "source_length"},
                "src_mask": {0: "batch", 1: "source_length"},
                "tgt_mask": {0: "batch", 1: "target_length"},
                "logits": {0: "batch", 1: "target_length"},
            },
            **common,
        )
        for path in (temp_encoder, temp_decoder):
            onnx.checker.check_model(onnx.load(path))
        shutil.copy2(temp_encoder, encoder_path)
        shutil.copy2(temp_decoder, decoder_path)
    for path in output_dir.glob("*.onnx.data"):
        path.unlink()

    for path in (encoder_path, decoder_path):
        graph = onnx.load(path)
        onnx.checker.check_model(graph)

    encoder_session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    decoder_session = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
    ort_memory = encoder_session.run(
        ["memory"], {"src_ids": src_ids.numpy(), "src_mask": src_mask.numpy()}
    )[0]
    with torch.inference_mode():
        torch.testing.assert_close(torch.from_numpy(ort_memory), memory, rtol=1e-4, atol=1e-5)
        expected_logits = model.decode(
            tgt_ids, torch.from_numpy(ort_memory), src_mask, tgt_mask=tgt_mask
        )
    ort_logits = decoder_session.run(
        ["logits"],
        {
            "tgt_ids": tgt_ids.numpy(),
            "memory": ort_memory,
            "src_mask": src_mask.numpy(),
            "tgt_mask": tgt_mask.numpy(),
        },
    )[0]
    torch.testing.assert_close(
        torch.from_numpy(ort_logits), expected_logits, rtol=1e-4, atol=1e-4
    )

    # Confirm both graphs accept sequence lengths different from the export examples.
    longer_src = torch.randint(0, cfg.model.vocab, (1, 7), dtype=torch.long)
    longer_src_mask = torch.ones(1, 7, dtype=torch.bool)
    longer_memory = encoder_session.run(
        ["memory"],
        {"src_ids": longer_src.numpy(), "src_mask": longer_src_mask.numpy()},
    )[0]
    longer_tgt = torch.randint(0, cfg.model.vocab, (1, 3), dtype=torch.long)
    longer_tgt_mask = torch.ones(1, 3, dtype=torch.bool)
    decoder_session.run(
        ["logits"],
        {
            "tgt_ids": longer_tgt.numpy(),
            "memory": longer_memory,
            "src_mask": longer_src_mask.numpy(),
            "tgt_mask": longer_tgt_mask.numpy(),
        },
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    state_path = checkpoint / "model.pt" if checkpoint.is_dir() else checkpoint
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    output_dir = Path(args.output_dir) if args.output_dir else state_path.parent / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("encoder.onnx", "decoder.onnx"):
        (output_dir / name).unlink(missing_ok=True)
    for path in output_dir.glob("*.onnx.data"):
        path.unlink()

    tokenizer = load_tokenizer(cfg.tokenizer.sp_model, max_len=cfg.model.max_len)
    validate_vocab(tokenizer, cfg.model.vocab)
    model = build_model(cfg.model).cpu().eval()
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    replace_rms_norms(model)
    original_attention = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = export_attention
    try:
        export_graphs(cfg, model, output_dir, args.opset)
    finally:
        F.scaled_dot_product_attention = original_attention
    shutil.copy2(cfg.tokenizer.sp_model, output_dir / "tokenizer.model")
    metadata = {
        "architecture": cfg.model.type,
        "vocab_size": cfg.model.vocab,
        "max_source_length": cfg.model.max_len,
        "max_target_length": cfg.gen.max_len,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "pad_id": tokenizer.pad_id,
        "opset": args.opset,
        "decoder_cache": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exported and verified ONNX graphs in {output_dir}")


if __name__ == "__main__":
    main()
