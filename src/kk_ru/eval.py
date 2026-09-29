from __future__ import annotations

import argparse
import json
import urllib.error
from pathlib import Path

import torch

from .config import load_config
from .data import iter_pairs
from .model import build_model
from .tokenizer import load_tokenizer, validate_vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KK→RU model on FLORES+.")
    parser.add_argument("--config", type=str, default="configs/p0.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--split", type=str, default="devtest", choices=["dev", "devtest"])
    parser.add_argument("--opts", nargs="*", default=None)
    parser.add_argument("--comet-checkpoint", default=None)
    parser.add_argument("--spbleu", action="store_true", help="compute FLORES spBLEU; may download a tokenizer")
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

    cache = None
    for step in range(max_len):
        tokens = seqs[:, :, -1:] if cache is not None else seqs
        logits, cache = model.decode(
            tokens.reshape(bsz * beam, -1),
            memory,
            src_mask_beams,
            past_key_values=cache,
            use_cache=True,
        )
        logp = torch.log_softmax(logits[:, -1, :], dim=-1).view(bsz, beam, vocab)
        logp = logp.masked_fill(finished.unsqueeze(-1), -float("inf"))
        logp[:, :, eos_id] = torch.where(finished, 0.0, logp[:, :, eos_id])

        cand = scores.unsqueeze(-1) + logp
        top_scores, top_idx = cand.view(bsz, -1).topk(beam, dim=-1)
        prev_beam = top_idx // vocab
        token = top_idx % vocab

        current_len = seqs.shape[-1]
        gathered = seqs.gather(1, prev_beam.unsqueeze(-1).expand(bsz, beam, current_len))
        seqs = torch.cat([gathered, token.unsqueeze(-1)], dim=-1)
        flat_indices = (prev_beam + torch.arange(bsz, device=device).unsqueeze(1) * beam).reshape(-1)
        cache = [(keys.index_select(0, flat_indices), values.index_select(0, flat_indices)) for keys, values in cache]
        scores = top_scores
        finished = finished.gather(1, prev_beam) | (token == eos_id)
        if finished.all():
            break

    lengths = (seqs != pad_id).sum(dim=-1).clamp_min(1).float()
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
    max_len: int,
    source_max_len: int,
    batch_size: int = 32,
) -> list[str]:
    model.eval()
    pad_id = tokenizer.pad_id
    hypotheses = []

    for i in range(0, len(sources), batch_size):
        chunk = sources[i : i + batch_size]
        encoded = [tokenizer.encode(s, add_bos=True)[:source_max_len] for s in chunk]
        longest = max(len(x) for x in encoded)
        src_ids = torch.full((len(chunk), longest), pad_id, dtype=torch.long, device=device)
        src_mask = torch.zeros((len(chunk), longest), dtype=torch.bool, device=device)
        for j, ids in enumerate(encoded):
            src_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            src_mask[j, : len(ids)] = True

        seqs = beam_search(
            model, src_ids, src_mask, tokenizer.bos_id, tokenizer.eos_id,
            pad_id, max_len, beam,
        )
        for seq in seqs:
            hypotheses.append(tokenizer.decode(seq, skip_special=True).strip())
    return hypotheses


def compute_metrics(
    hypotheses: list[str],
    references: list[str],
    sources: list[str] | None = None,
    comet_checkpoint: str | None = None,
    compute_spbleu: bool = False,
) -> dict:
    import sacrebleu

    metrics = {
        "bleu": sacrebleu.corpus_bleu(hypotheses, [references]).score,
        "chrf": sacrebleu.corpus_chrf(hypotheses, [references]).score,
        "chrf++": sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score,
    }
    if compute_spbleu:
        try:
            metrics["spbleu"] = sacrebleu.corpus_bleu(
                hypotheses, [references], tokenize="flores200"
            ).score
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"warning: spBLEU skipped because FLORES tokenizer is unavailable: {error}")
    if comet_checkpoint is not None:
        if sources is None:
            raise ValueError("COMET requires source sentences")
        from comet import load_from_checkpoint

        comet_model = load_from_checkpoint(comet_checkpoint)
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
        metrics["comet"] = comet_model.predict(
            data, batch_size=8, gpus=int(torch.cuda.is_available())
        )["system_score"]
    return metrics


def evaluate(
    cfg, tokenizer, model, split: str, device: torch.device,
    write: bool = False, comet_checkpoint: str | None = None,
    compute_spbleu: bool = False,
) -> dict:
    paths = {"dev": cfg.data.eval_dev, "devtest": cfg.data.eval_test}
    path = paths[split]
    sources, references = [], []
    for kk, ru in iter_pairs(path):
        sources.append(kk)
        references.append(ru)
    hypotheses = translate(
        model,
        tokenizer,
        sources,
        device,
        cfg.gen.beam,
        cfg.gen.max_len,
        source_max_len=cfg.model.max_len,
    )
    metrics = compute_metrics(
        hypotheses, references, sources, comet_checkpoint, compute_spbleu
    )
    if write:
        out = Path(cfg.paths.reports) / f"{split}.hyp.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(hypotheses), encoding="utf-8")
        (out.parent / f"{split}.metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.opts)
    device = _device()

    tokenizer = load_tokenizer(cfg.tokenizer.sp_model, max_len=cfg.model.max_len)
    validate_vocab(tokenizer, cfg.model.vocab)
    model = build_model(cfg.model)
    checkpoint = Path(args.checkpoint)
    state_path = checkpoint / "model.pt" if checkpoint.is_dir() else checkpoint
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    metrics = evaluate(
        cfg, tokenizer, model, args.split, device,
        write=True, comet_checkpoint=args.comet_checkpoint,
        compute_spbleu=args.spbleu,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.2f}")


if __name__ == "__main__":
    main()
