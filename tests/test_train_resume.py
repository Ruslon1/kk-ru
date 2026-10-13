from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from kk_ru.train import _phase_start, _reset_scheduler, build_scheduler


class TrainResumeTests(unittest.TestCase):
    def test_new_data_phase_resets_cursor_but_keeps_global_step(self):
        state = {"step": 1200, "epoch": 1, "batch_index": 450, "phase_step": 1200, "phase_total_steps": 5000, "phase": "original"}
        phase = _phase_start(state, 9000, resume_new_data=True)
        self.assertEqual(phase["global_step"], 1200)
        self.assertEqual(phase["phase_step"], 0)
        self.assertEqual(phase["phase_total_steps"], 9000)
        self.assertEqual(phase["epoch"], 0)
        self.assertEqual(phase["batch_index"], 0)
        self.assertTrue(phase["reset_scheduler"])

    def test_exact_resume_keeps_checkpoint_cursor_and_phase_progress(self):
        state = {"step": 1500, "epoch": 2, "batch_index": 77, "phase_step": 500, "phase_total_steps": 9000, "phase": "merged_data"}
        phase = _phase_start(state, 9000, resume_new_data=False)
        self.assertEqual(phase["phase_step"], 500)
        self.assertEqual(phase["phase_total_steps"], 9000)
        self.assertEqual(phase["epoch"], 2)
        self.assertEqual(phase["batch_index"], 77)
        self.assertFalse(phase["reset_scheduler"])

    def test_new_data_schedule_restarts_without_resetting_optimizer_state(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        parameter.grad = torch.tensor(1.0)
        optimizer.step()
        optimizer_state = optimizer.state[parameter]["step"].clone()
        config = SimpleNamespace(lr=0.1, warmup_steps=2, min_lr=0.01)
        old_scheduler = build_scheduler(optimizer, config, total_steps=10)
        optimizer.step()
        old_scheduler.step()
        optimizer_state = optimizer.state[parameter]["step"].clone()

        scheduler = _reset_scheduler(None, optimizer, old_scheduler, config, total_steps=30)

        self.assertEqual(optimizer.state[parameter]["step"], optimizer_state)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(scheduler.state_dict()["lr_lambdas"], [None])
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)


if __name__ == "__main__":
    unittest.main()
