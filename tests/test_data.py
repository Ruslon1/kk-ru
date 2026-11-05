from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import sentencepiece as spm

from kk_ru.data import TranslationDataset
from kk_ru.tokenizer import load_tokenizer


class TranslationDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "corpus.tmp-import"
        root.mkdir()
        self.path = root / "pairs.tsv"
        self.path.write_text(
            "Бұл қазақ мәтіні.\tЭто русский текст.\n"
            "Бұл ұзын қазақ сөйлемі.\tЭто более длинное русское предложение.\n",
            encoding="utf-8",
        )
        spm.SentencePieceTrainer.train(
            input=str(self.path), model_prefix=str(root / "tokenizer"),
            vocab_size=64, hard_vocab_limit=False, pad_id=3, minloglevel=2,
        )
        self.tokenizer = load_tokenizer(str(root / "tokenizer.model"))
        self.cache = Path(f"{self.path}.tok")

    def test_disk_and_limited_datasets_use_identical_tokens_and_eos(self):
        disk = TranslationDataset(str(self.path), self.tokenizer, max_len=8)
        limited = TranslationDataset(str(self.path), self.tokenizer, max_len=8, limit=1)
        reused = TranslationDataset(str(self.path), self.tokenizer, max_len=8, cache=False)

        self.assertEqual(len(disk), 2)
        self.assertEqual(len(limited), 1)
        self.assertEqual(disk[0], limited[0])
        rows = [disk[index] for index in range(len(disk))]
        self.assertEqual(rows, [reused[index] for index in range(len(reused))])
        np.testing.assert_array_equal(disk.lengths(), [sum(map(len, row)) for row in rows])
        for source, target in rows:
            self.assertEqual(source[0], self.tokenizer.bos_id)
            self.assertEqual(target[0], self.tokenizer.bos_id)
            self.assertEqual(target[-1], self.tokenizer.eos_id)
            self.assertLessEqual(len(target), 8)
        self.assertTrue((self.cache / "meta.json").is_file())
        self.assertEqual(self.tokenizer.encode(""), [])

    def test_stale_cache_is_rejected_or_rebuilt_after_source_changes(self):
        TranslationDataset(str(self.path), self.tokenizer, max_len=8)
        with self.path.open("a", encoding="utf-8") as file:
            file.write("Мәтін.\tТекст.\n")

        with self.assertRaisesRegex(ValueError, "stale or incompatible"):
            TranslationDataset(str(self.path), self.tokenizer, max_len=8, cache=False)

        rebuilt = TranslationDataset(str(self.path), self.tokenizer, max_len=8)
        self.assertEqual(len(rebuilt), 3)

    def test_incomplete_cache_is_rejected_or_rebuilt(self):
        self.cache.mkdir()
        (self.cache / "src.bin").write_bytes(b"")

        with self.assertRaisesRegex(FileNotFoundError, "token cache is missing"):
            TranslationDataset(str(self.path), self.tokenizer, max_len=8, cache=False)

        rebuilt = TranslationDataset(str(self.path), self.tokenizer, max_len=8)
        self.assertEqual(len(rebuilt), 2)
