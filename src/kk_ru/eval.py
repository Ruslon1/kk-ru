from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import load_config
from .data import iter_pairs
from .model import build_model
from .tokenizer import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KK→RU model on FLORES+.")
    parser.add_argument("--config", type=str, default="configs/p0.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--split", type=str, default="devtest", choices=["dev", "devtest"])
    parser.add_argument("--opts", nargs="*", default=None)
    return parser.parse_args()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def beam_search(
    model,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    beam: int,
    alpha: float = 0.6,
) -> list[list[int]]:
    model.eval()
    bsz = src_ids.shape[0]
    device = src_ids.device
    vocab = model.embed_tokens.weight.shape[0]

    memory = model.encode(src_ids, src_mask)
    s_len, d = memory.shape[1], memory.shape[2]
    memory = memory.unsqueeze(1).expand(bsz, beam, s_len, d).reshape(bsz * beam, s_len, d)
    src_mask_beams = src_mask.unsqueeze(1).expand(bsz, beam, s_len).reshape(bsz * beam, s_len)

    seqs = torch.full((bsz, beam, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.full((bsz, beam), -float("inf"), device=device)
    scores[:, 0] = 0.0
    finished = torch.zeros(bsz, beam, dtype=torch.bool, device=device)

    for _ in range(max_len):
        cur = seqs.shape[-1]
        logits = model.decode(seqs.reshape(bsz * beam, cur), memory, src_mask_beams)
        logp = torch.log_softmax(logits[:, -1, :], dim=-1).view(bsz, beam, vocab)
        logp = logp.masked_fill(finished.unsqueeze(-1), -float("inf"))
        logp[:, :, pad_id] = torch.where(finished, 0.0, logp[:, :, pad_id])

        cand = scores.unsqueeze(-1) + logp
        top_scores, top_idx = cand.view(bsz, -1).topk(beam, dim=-1)
        prev_beam = top_idx // vocab
        token = top_idx % vocab

        gathered = seqs.gather(1, prev_beam.unsqueeze(-1).expand(bsz, beam, cur))
        seqs = torch.cat([gathered, token.unsqueeze(-1)], dim=-1)
        scores = top_scores
        finished = finished.gather(1, prev_beam) | (token == eos_id)

    lengths = (seqs != pad_id).sum(dim=-1).float()
    norm = scores / lengths.pow(alpha)
    best = norm.argmax(dim=-1)
    return seqs[torch.arange(bsz), best].tolist()


@torch.no_grad()
def translate(
    model,
    tokenizer,
    sources: list[str],
    device: torch.device,
    beam: int,
    batch_size: int = 32,
) -> list[str]:
    model.eval()
    pad_id = tokenizer.pad_id
    max_len = tokenizer.max_len
    hypotheses = []

    for i in range(0, len(sources), batch_size):
        chunk = sources[i : i + batch_size]
        encoded = [tokenizer.encode(s, add_bos=True)[:max_len] for s in chunk]
        longest = max(len(x) for x in encoded)
        src_ids = torch.full((len(chunk), longest), pad_id, dtype=torch.long, device=device)
        src_mask = torch.zeros((len(chunk), longest), dtype=torch.bool, device=device)
        for j, ids in enumerate(encoded):
            src_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            src_mask[j, : len(ids)] = True

        seqs = beam_search(model, src_ids, src_mask, tokenizer.bos_id, tokenizer.eos_id, pad_id, max_len, beam)
        for seq in seqs:
            hypotheses.append(tokenizer.decode(seq, skip_special=True).strip())
    return hypotheses


def compute_metrics(hypotheses: list[str], references: list[str], sources: list[str] | None = None) -> dict:
    import sacrebleu

    metrics = {
        "bleu": sacrebleu.corpus_bleu(hypotheses, [references]).score,
        "chrf": sacrebleu.corpus_chrf(hypotheses, [references]).score,
    }
    if sources is not None:
        try:
            from comet import download_model, load_from_checkpoint

            model_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(model_path)
            data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
            metrics["comet"] = comet_model.predict(data, batch_size=8, gpus=1)["system_score"]
        except Exception:
            pass
    return metrics


def evaluate(cfg, tokenizer, model, split: str, device: torch.device, write: bool = False) -> dict:
    path = cfg.data.eval_dev if split == "dev" else cfg.data.eval_test
    sources, references = [], []
    for kk, ru in iter_pairs(path):
        sources.append(kk)
        references.append(ru)
    hypotheses = translate(model, tokenizer, sources, device, cfg.gen.beam)
    metrics = compute_metrics(hypotheses, references, sources)
    if write:
        out = Path(cfg.paths.reports) / f"{split}.hyp.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(hypotheses), encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.opts)
    device = _device()

    tokenizer = load_tokenizer(cfg.tokenizer.sp_model, max_len=cfg.model.max_len)
    model = build_model(cfg.model)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    metrics = evaluate(cfg, tokenizer, model, args.split, device, write=True)
    for name, value in metrics.items():
        print(f"{name}: {value:.2f}")


if __name__ == "__main__":
    main()
