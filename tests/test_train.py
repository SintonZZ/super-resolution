import logging
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from archs import build_model
from losses import GANLoss
from train import (
    ModelEMA,
    build_checkpoint,
    get_step_lr,
    load_pretrained_model,
    resume_training,
    train_stage2_step,
    update_best_metric,
    validate_initialization_config,
    validate_training_config,
)
from util import extract_state_dict


class StepTrainingTest(unittest.TestCase):
    def test_stage2_step_updates_generator_discriminator_and_ema(self):
        class TinyPerceptual(nn.Module):
            def forward(self, prediction, target):
                return F.l1_loss(prediction, target), prediction.new_zeros(())

        generator = nn.Conv2d(3, 3, 1)
        discriminator = nn.Conv2d(3, 1, 1)
        ema = ModelEMA(generator, decay=0.5)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
        generator_before = generator.weight.detach().clone()
        discriminator_before = discriminator.weight.detach().clone()
        metrics = train_stage2_step(
            generator,
            discriminator,
            {
                "input": torch.rand(1, 3, 8, 8),
                "target": torch.rand(1, 3, 8, 8),
            },
            nn.L1Loss(),
            1.0,
            TinyPerceptual(),
            GANLoss(),
            0.1,
            optimizer_g,
            optimizer_d,
            scaler=None,
            ema=ema,
            use_amp=False,
            device=torch.device("cpu"),
            stage_config={
                "total_steps": 2,
                "max_lr": 1e-4,
                "scheduler": {"type": "multistep", "milestones": [2]},
            },
            step=0,
            usm_sharpener=None,
        )
        self.assertFalse(torch.equal(generator.weight, generator_before))
        self.assertFalse(torch.equal(discriminator.weight, discriminator_before))
        self.assertIn("perceptual_loss", metrics)
        self.assertIn("d_loss", metrics)

    def test_pretrained_prefers_ema_and_starts_a_fresh_ema(self):
        online = nn.Linear(1, 1, bias=False)
        ema_source = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            online.weight.fill_(2.0)
            ema_source.weight.fill_(1.0)
        checkpoint = {
            "model_state_dict": online.state_dict(),
            "params_ema": ema_source.state_dict(),
            "optimizer_state_dict": {"should_not_be_loaded": True},
            "step": 999,
            "best_metric": 42.0,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrained.pth"
            torch.save(checkpoint, path)
            target = nn.Linear(1, 1, bias=False)
            state_key = load_pretrained_model(
                target,
                path,
                "cpu",
                logging.getLogger("test_pretrained"),
            )
            fresh_optimizer = torch.optim.Adam(target.parameters(), lr=1e-5)
            fresh_ema = ModelEMA(target, decay=0.999)

        self.assertEqual(state_key, "params_ema")
        self.assertAlmostEqual(target.weight.item(), 1.0)
        self.assertAlmostEqual(fresh_ema.module.weight.item(), 1.0)
        self.assertEqual(fresh_optimizer.state_dict()["state"], {})

    def test_resume_and_pretrained_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_initialization_config({
                "resume": "latest_model.pth",
                "pretrained": "best_model.pth",
            })

    def test_step_lr_uses_linear_warmup_then_cosine_decay(self):
        config = {
            "total_steps": 100,
            "warmup_steps": 10,
            "max_lr": 1e-3,
            "min_lr": 1e-5,
        }
        self.assertAlmostEqual(get_step_lr(0, config), 1e-4)
        self.assertAlmostEqual(get_step_lr(9, config), 1e-3)
        self.assertAlmostEqual(get_step_lr(10, config), 1e-3)
        self.assertAlmostEqual(get_step_lr(99, config), 1e-5)

    def test_multistep_lr_stays_constant_until_milestone(self):
        config = {
            "total_steps": 100,
            "warmup_steps": 0,
            "max_lr": 2e-4,
            "scheduler": {
                "type": "multistep",
                "milestones": [100],
                "gamma": 0.5,
            },
        }
        self.assertAlmostEqual(get_step_lr(0, config), 2e-4)
        self.assertAlmostEqual(get_step_lr(99, config), 2e-4)

    def test_ema_updates_parameters_and_is_preferred_for_inference(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
        ema = ModelEMA(model, decay=0.5)
        with torch.no_grad():
            model.weight.fill_(2.0)
        ema.update(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        checkpoint = build_checkpoint(
            model,
            ema,
            optimizer,
            scaler=None,
            step=7,
            data_epoch=2,
            best_metric=31.5,
            metrics={"train": None, "val": None},
            config={},
        )
        state_dict, key = extract_state_dict(checkpoint)

        self.assertEqual(key, "params_ema")
        self.assertAlmostEqual(state_dict["weight"].item(), 1.0)
        self.assertAlmostEqual(checkpoint["model_state_dict"]["weight"].item(), 2.0)

    def test_spanf_ema_refreshes_reparameterized_evaluation_convs(self):
        model = build_model(
            {
                "type": "spanf",
                "in_channels": 3,
                "out_channels": 3,
                "feature_channels": 4,
                "upscale": 2,
                "bias": True,
                "nearest_init": True,
            }
        )
        ema = ModelEMA(model, decay=0.0)
        inputs = torch.rand(1, 3, 8, 8)

        ema.module.eval()
        ema.module(inputs)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(0.01)
        model.train()
        model(inputs)
        ema.update(model)

        model.eval()
        expected = model(inputs)
        actual = ema.module(inputs)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-4))

    def test_resume_restores_step_online_model_and_ema(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ema = ModelEMA(model, decay=0.5)
        with torch.no_grad():
            model.weight.fill_(2.0)
        ema.update(model)
        checkpoint = build_checkpoint(
            model,
            ema,
            optimizer,
            scaler=None,
            step=7,
            data_epoch=2,
            best_metric=31.5,
            metrics={"train": None, "val": None},
            config={},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save(checkpoint, path)
            restored_model = nn.Linear(1, 1, bias=False)
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
            restored_ema = ModelEMA(restored_model, decay=0.5)
            step, data_epoch, best_metric = resume_training(
                restored_model,
                restored_ema,
                restored_optimizer,
                scaler=None,
                resume_path=path,
                device="cpu",
                logger=logging.getLogger("test_resume_training"),
                steps_per_epoch=5,
            )

        self.assertEqual(step, 7)
        self.assertEqual(data_epoch, 2)
        self.assertAlmostEqual(best_metric, 31.5)
        self.assertAlmostEqual(restored_model.weight.item(), 2.0)
        self.assertAlmostEqual(restored_ema.module.weight.item(), 1.0)

    def test_stage2_checkpoint_restores_discriminator_and_both_optimizers(self):
        model = nn.Linear(1, 1, bias=False)
        discriminator = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.0)
            discriminator.weight.fill_(3.0)
        ema = ModelEMA(model, decay=0.5)
        optimizer_g = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
        checkpoint = build_checkpoint(
            model,
            ema,
            optimizer_g,
            scaler=None,
            step=9,
            data_epoch=1,
            best_metric=30.0,
            metrics={},
            config={},
            active_stage="stage2",
            completed_stages=["stage1"],
            discriminator=discriminator,
            optimizer_d=optimizer_d,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage2.pth"
            torch.save(checkpoint, path)
            restored_model = nn.Linear(1, 1, bias=False)
            restored_discriminator = nn.Linear(1, 1, bias=False)
            restored_ema = ModelEMA(restored_model)
            restored_g_optimizer = torch.optim.Adam(
                restored_model.parameters(), lr=1e-4
            )
            restored_d_optimizer = torch.optim.Adam(
                restored_discriminator.parameters(), lr=1e-4
            )
            step, _, _ = resume_training(
                restored_model,
                restored_ema,
                restored_g_optimizer,
                scaler=None,
                resume_path=path,
                device="cpu",
                logger=logging.getLogger("test_stage2_resume"),
                steps_per_epoch=1,
                discriminator=restored_discriminator,
                optimizer_d=restored_d_optimizer,
            )

        self.assertEqual(step, 9)
        self.assertAlmostEqual(restored_model.weight.item(), 2.0)
        self.assertAlmostEqual(restored_discriminator.weight.item(), 3.0)

    def test_best_metric_only_changes_after_validation(self):
        best_metric, is_best = update_best_metric(None, 30.0, "y_psnr")
        self.assertEqual(best_metric, 30.0)
        self.assertFalse(is_best)

        best_metric, is_best = update_best_metric(
            {"rgb_psnr": 31.0, "y_psnr": 32.0},
            best_metric,
            "y_psnr",
        )
        self.assertEqual(best_metric, 32.0)
        self.assertTrue(is_best)

    def test_lower_is_better_metric(self):
        best_metric, is_best = update_best_metric(
            {"lpips": 0.2},
            0.3,
            "lpips",
            mode="min",
        )
        self.assertEqual(best_metric, 0.2)
        self.assertTrue(is_best)

    def test_two_stage_config_requires_pretrained_when_stage1_is_disabled(self):
        config = {
            "training": {
                "stage1": {"enabled": False},
                "stage2": {
                    "enabled": True,
                    "total_steps": 10,
                    "batch_size": 1,
                    "loss": {"pixel": {"type": "l1"}},
                },
            },
            "discriminator": {"type": "unet_sn"},
        }
        with self.assertRaisesRegex(ValueError, "pretrained_generator"):
            validate_training_config(config)


if __name__ == "__main__":
    unittest.main()
