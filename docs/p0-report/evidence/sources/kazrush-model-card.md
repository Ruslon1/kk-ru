---
library_name: transformers
pipeline_tag: translation
tags:
- transformers
- translation
- pytorch
- russian
- kazakh

license: apache-2.0
language:
- ru
- kk
datasets:
- issai/kazparc
---

# kazRush-kk-ru

kazRush-kk-ru is a translation model for translating from Kazakh to Russian. The model was trained with randomly initialized weights based on the T5 configuration on the available open-source parallel data.

## Usage

Using the model requires `sentencepiece` library to be installed.

After installing necessary dependencies the model can be run with the following code:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained('deepvk/kazRush-kk-ru').to(device)
tokenizer = AutoTokenizer.from_pretrained('deepvk/kazRush-kk-ru')

@torch.inference_mode
def generate(text, **kwargs):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    hypotheses = model.generate(**inputs, num_beams=5, **kwargs)
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)

print(generate("Анам жақтауды жуды."))
```

You can also access the model via _pipeline_ wrapper:
```python
>>> from transformers import pipeline

>>> pipe = pipeline(model="deepvk/kazRush-kk-ru")
>>> pipe("Иттерді кім шығарды?")
[{'translation_text': 'Кто выпустил собак?'}]
```

## Data and Training

This model was trained on the following data (Russian-Kazakh language pairs):

| Dataset  | Number of pairs |
|-----------------------------------------|-------|
| [OPUS Corpora](<https://opus.nlpl.eu/results/ru&kk/corpus-result-table>)     | 718K   |
| [kazparc](<https://huggingface.co/datasets/issai/kazparc>)            | 2,150K   |
| [wmt19 dataset](<https://statmt.org/wmt19/translation-task.html#download>)                  | 5,063K   |
| [TIL dataset](<https://github.com/turkic-interlingua/til-mt/tree/master/til_corpus>)                  | 4,403K   |

Preprocessing of the data included:
1. deduplication
2. removing trash symbols, special tags, multiple whitespaces etc. from texts
3. removing texts that were not in Russian or Kazakh (language detection was made via [facebook/fasttext-language-identification](<https://huggingface.co/facebook/fasttext-language-identification>))
4. removing pairs that had low alingment score (comparison was performed via [sentence-transformers/LaBSE](<https://huggingface.co/sentence-transformers/LaBSE>))
5. filtering the data using [opusfilter](<https://github.com/Helsinki-NLP/OpusFilter>) tools

The model was trained for 56 hours on 2 GPUs NVIDIA A100 80 Gb.

## Evaluation

Current model was compared to another open-source translation model, [NLLB](<https://huggingface.co/docs/transformers/model_doc/nllb>). We compared our model to all version of NLLB, excluding nllb-moe-54b due to its size.
The metrics - BLEU, chrF and COMET - were calculated on `devtest` part of [FLORES+ evaluation benchmark](<https://github.com/openlanguagedata/flores>), most recent evaluation benchmark for multilingual machine translation.
Calculation of BLEU and chrF follows the standart implementation from [sacreBLEU](<https://github.com/mjpost/sacrebleu>), and COMET is calculated using default model described in [COMET repository](<https://github.com/Unbabel/COMET>).

| Model  | Size | BLEU | chrf | COMET |
|-----------------------------------------|-------|-----------------------------|------------------------|----------|
| [nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)    | 600M   | 18.0  |  47.3  | 85.6 |
| This model                             | 197M    | 18.8      | 48.7                   | 86.7 |
| [nllb-200-1.3B](https://huggingface.co/facebook/nllb-200-1.3B)                   | 1.3B   | 20.4 | 49.3          | 87.9 |
| [nllb-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B)   | 1.3B    | 20.8 | 49.6 | 88.1 |
| [nllb-200-3.3B](https://huggingface.co/facebook/nllb-200-3.3B)                    | 3.3B    | **21.5**  | **50.7**  | **88.7** |

## Examples of usage:

```python
>>> print(generate("Балық көбінесе сулардағы токсиндердің жоғары концентрацияларына байланысты өледі."))
Рыба часто умирает из-за высоких концентраций токсинов в воде.

>>> print(generate("Өткен 3 айда 80-нен астам қамалушы ресми түрде айып тағылмастан изолятордан шығарылды."))
За прошедшие 3 месяца более 80 арестованных были официально извлечены из изолятора без обвинения.

>>> print(generate("Бұл тастардың он бесі өткен шілде айындағы метеориттік жаңбырға жатқызылады."))
Пятнадцать этих камней относят к метеоритным дождям прошлого июля.
```

## Citations

```
@misc{deepvk2024kazRushkkru,
    title={kazRush-kk-ru: translation model from Kazakh to Russian},
    author={Lebedeva, Anna and  Sokolov, Andrey},
    url={https://huggingface.co/deepvk/kazRush-kk-ru},
    publisher={Hugging Face},
    year={2024},
}
```
