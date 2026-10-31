"""Generate report tables and paired-bootstrap evidence from saved translations."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

MODEL_URLS = {
    "nllb-200-distilled-600M": "https://huggingface.co/facebook/nllb-200-distilled-600M",
    "nllb-200-1.3B": "https://huggingface.co/facebook/nllb-200-1.3B",
    "nllb-200-distilled-1.3B": "https://huggingface.co/facebook/nllb-200-distilled-1.3B",
    "nllb-200-3.3B": "https://huggingface.co/facebook/nllb-200-3.3B",
    "KazRush KK-RU": "https://huggingface.co/deepvk/kazRush-kk-ru",
    "Qwen3-0.6B": "https://huggingface.co/Qwen/Qwen3-0.6B",
    "qwen/qwen3.8-27b": "https://openrouter.ai/qwen/qwen3.8-27b",
    "google/gemma-4-26b-a4b-it": "https://openrouter.ai/google/gemma-4-26b-a4b-it",
    "google/gemma-4-31b-it": "https://openrouter.ai/google/gemma-4-31b-it",
    "Google Translate": "https://translate.google.com/",
}


def read(name):
    return json.loads((HERE / "evidence" / name).read_text())


def tex(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in text)


def main():
    generated = HERE / "generated"
    generated.mkdir(exist_ok=True)
    p0 = read("p0-devtest.json")
    archives = read("archived_benchmarks.json")["results"]
    kaz = next(row for row in archives if row["model"] == "KazRush KK-RU")
    rows = [json.loads(line) for line in (ROOT / "reports/p0-report/p0-devtest.jsonl").read_text().splitlines()]
    references = [r["reference"] for r in rows]
    hypotheses = [r["hypothesis"] for r in rows]
    kaz_hypotheses = (ROOT / "reports/kazRush-devtest.hyp.txt").read_text().splitlines()
    os.environ["SACREBLEU_SEED"] = "12345"
    signatures, scores = PairedTest(
        [("KazRush", kaz_hypotheses), ("P0", hypotheses)],
        {"BLEU": BLEU(), "chrF++": CHRF(word_order=2)}, [references],
        test_type="bs", n_samples=1000, n_jobs=1,
    )()
    bootstrap = {"method": "SacreBLEU paired bootstrap", "samples": 1000, "seed": 12345,
                 "alignment_assumption": "KazRush line i corresponds to devtest row i",
                 "signatures": {k: str(v) for k, v in signatures.items()},
                 "results": {k: [vars(v) if hasattr(v, "__dict__") else v for v in values]
                             for k, values in scores.items()}}
    (HERE / "evidence/paired_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, default=lambda value: value.item()) + "\n")

    macros = {"PZeroBLEU": f"{p0['bleu']:.2f}", "PZeroChrF": f"{p0['chrf']:.2f}",
              "PZeroChrFPP": f"{p0['chrf++']:.2f}", "PZeroTER": f"{p0['ter']:.2f}",
              "KazBLEU": f"{kaz['bleu']:.2f}", "KazChrF": f"{kaz['chrf']:.2f}",
              "KazChrFPP": f"{kaz['chrf++']:.2f}", "KazTER": f"{kaz['ter']:.2f}",
              "DeltaBLEU": f"{p0['bleu']-kaz['bleu']:.2f}",
              "DeltaChrFPP": f"{p0['chrf++']-kaz['chrf++']:.2f}",
              "DeltaTER": f"{kaz['ter']-p0['ter']:.2f}",
              "EvalSeconds": f"{p0['inference_seconds_including_first_batch']:.1f}",
              "EvalThroughput": f"{p0['sentences_per_second']:.2f}"}
    (generated / "numbers.tex").write_text("\n".join(
        rf"\newcommand{{\{key}}}{{{value}}}" for key, value in macros.items()) + "\n")

    table = []
    for row in [p0, kaz, *[r for r in archives if r is not kaz]]:
        name = "P0 (fresh evaluation)" if row is p0 else row["model"]
        values = [tex(name)] + [f"{row[key]:.2f}" for key in ("bleu", "chrf", "chrf++", "ter")]
        table.append(" & ".join(values) + r" \\")
    (generated / "all_benchmarks.tex").write_text("\n".join(table) + "\n")

    names = {
        "Qwen3-0.6B": ("Qwen3-0.6B", "600M"),
        "qwen/qwen3.8-27b": ("Qwen3.8-27B", "27B"),
        "google/gemma-4-26b-a4b-it": ("Gemma-4-26B-A4B-IT", "26B"),
        "google/gemma-4-31b-it": ("Gemma-4-31B-IT", "31B"),
    }
    published = {r["model"]: r for r in read("published_benchmarks.json")["results"]}
    local = [p0, *archives]
    local_names = {r["model"] for r in local}
    combined = local + [r for name, r in published.items() if name not in local_names]
    table = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
             r"\toprule Model & Size & BLEU $\uparrow$ & chrF $\uparrow$ & chrF++ $\uparrow$ & TER $\downarrow$ & COMET $\uparrow$\\",
             r"\midrule"]
    for row in sorted(combined, key=lambda r: r["bleu"], reverse=True):
        source = published.get(row["model"], {})
        name, size = names.get(row["model"], (row["model"], source.get("size", "---")))
        name = tex(name)
        if row["model"] in MODEL_URLS:
            name = rf"\href{{{MODEL_URLS[row['model']]}}}{{{name}}}"
        if row["model"] not in local_names:
            name += r"$^\dagger$"
        if row is p0:
            name, size = r"\textbf{P0 (ours)}", "202M"
        values = [f"{row[key]:.2f}" if key in row else "---"
                  for key in ("bleu", "chrf", "chrf++", "ter")]
        if row is p0:
            values = [r"\textbf{" + v + "}" for v in values]
        comet = f"{source['comet']:.1f}" if "comet" in source else "---"
        if row["model"] in local_names and "comet" in source:
            comet += r"$^\dagger$"
        table.append(" & ".join([name, size, *values, comet]) + r" \\")
    table += [r"\bottomrule", r"\end{tabular*}"]
    (generated / "comparison.tex").write_text("\n".join(table) + "\n")

    table = []
    for metric, values in scores.items():
        if metric == "System":
            continue
        a, b = values
        for system, value in [("KazRush", a), ("P0", b)]:
            p_value = "---" if value.p_value is None else f"{value.p_value:.4f}"
            table.append(f"{tex(metric)} & {system} & {value.score:.2f} & "
                         f"{value.mean:.2f} $\\pm$ {value.ci:.2f} & {p_value}" + r" \\")
    (generated / "bootstrap.tex").write_text("\n".join(table) + "\n")

    suite = ET.parse(HERE / "evidence/tests.xml")
    test_rows = []
    for case in suite.findall(".//testcase"):
        module = case.get("classname").split(".")[1]
        test_rows.append(r"\path{" + module + "} & \\path{" + case.get("name") + r"} & Pass \\")
    (generated / "test_inventory.tex").write_text("\n".join(test_rows) + "\n")

    examples = []
    selected = []
    for index in (0, 2):
        row = rows[index]
        selected.append({**row, "kazrush": kaz_hypotheses[index]})
        examples.append(r"\paragraph{Example " + str(index + 1) + r" (devtest row " + str(index + 1) + ").}")
        for label, value in [("Source", row["source"]), ("Reference", row["reference"]),
                             ("P0", row["hypothesis"]), ("KazRush", kaz_hypotheses[index])]:
            examples.append(r"\noindent\textbf{" + label + r".} {\examplefont " + tex(value) + r"}\par\smallskip")
    (generated / "examples.tex").write_text("\n".join(examples) + "\n")
    (HERE / "evidence/qualitative_examples.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(bootstrap, indent=2, default=lambda value: value.item()))


if __name__ == "__main__":
    main()
