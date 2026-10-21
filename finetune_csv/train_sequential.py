"""Train a tokenizer, a predictor, or both in chronological CSV experiments."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download

from model import Kronos, KronosTokenizer
from finetune_csv.config_loader import CustomFinetuneConfig
from finetune_csv.data_utils import load_candles, split_manifest
from finetune_csv.finetune_tokenizer import train_tokenizer, set_seed, setup_logging as tokenizer_logging
from finetune_csv.finetune_base_model import train_model, setup_logging as predictor_logging


def initialize_from_architecture(model_class, architecture_path):
    """Use only the architecture JSON; pretrained tensor weights are never loaded."""
    local = Path(architecture_path) / "config.json"
    path = local if local.is_file() else Path(hf_hub_download(architecture_path, "config.json"))
    architecture = json.loads(path.read_text())
    parameters = inspect.signature(model_class.__init__).parameters
    kwargs = {key: value for key, value in architecture.items() if key in parameters and key != "self"}
    missing = [key for key, param in parameters.items() if key != "self" and
               param.default is inspect.Parameter.empty and key not in kwargs]
    if missing:
        raise ValueError(f"Architecture config is missing required fields: {', '.join(missing)}")
    return model_class(**kwargs)


def checkpoint_exists(path):
    path = Path(path)
    return ((path / "config.json").is_file() and
            ((path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()))


class SequentialTrainer:
    def __init__(self, config_path=None):
        self.config = CustomFinetuneConfig(config_path)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.config.device_id)))
        if self.config.use_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
        self.config.print_config_summary()
        print(f"Using device: {self.device}; rank={self.rank}, world_size={self.world_size}")

    def _setup_distributed(self):
        if self.world_size > 1 and not dist.is_initialized():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            dist.init_process_group(backend=os.environ.get("DIST_BACKEND", backend))

    def _logger(self, tokenizer=False):
        setup = tokenizer_logging if tokenizer else predictor_logging
        return setup(self.config.exp_name, str(Path(self.config.base_save_path) / "logs"), self.rank)

    def train_tokenizer_phase(self):
        if self.config.skip_existing and checkpoint_exists(self.config.tokenizer_best_model_path):
            print("Tokenizer checkpoint exists; skipping tokenizer training")
            return True
        set_seed(self.config.seed)
        if self.config.pre_trained_tokenizer:
            tokenizer = KronosTokenizer.from_pretrained(self.config.pretrained_tokenizer_path)
        else:
            print("Randomly initializing tokenizer weights from architecture JSON")
            tokenizer = initialize_from_architecture(KronosTokenizer, self.config.pretrained_tokenizer_path)
        tokenizer = tokenizer.to(self.device)
        best_loss = train_tokenizer(tokenizer, self.device, self.config,
                                    self.config.tokenizer_save_path, self._logger(tokenizer=True))
        self._record_phase("tokenizer", best_loss, self.config.tokenizer_best_model_path)
        del tokenizer
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return True

    def train_basemodel_phase(self):
        if self.config.skip_existing and checkpoint_exists(self.config.basemodel_best_model_path):
            print("Predictor checkpoint exists; skipping predictor training")
            return True
        set_seed(self.config.seed)
        tokenizer_path = self.config.finetuned_tokenizer_path
        # A scratch predictor must reuse the tokenizer learned in stage one.
        if not tokenizer_path:
            raise ValueError("Set model_paths.finetuned_tokenizer to the frozen tokenizer checkpoint")
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(self.device)
        tokenizer.eval()
        tokenizer.requires_grad_(False)
        if self.config.pre_trained_predictor:
            model = Kronos.from_pretrained(self.config.pretrained_predictor_path)
        else:
            print("Randomly initializing predictor weights from architecture JSON")
            model = initialize_from_architecture(Kronos, self.config.pretrained_predictor_path)
        model = model.to(self.device)
        if (tokenizer.s1_bits, tokenizer.s2_bits) != (model.s1_bits, model.s2_bits):
            raise ValueError("Tokenizer codebook sizes do not match predictor architecture")
        print(f"Frozen tokenizer: {tokenizer_path}; predictor parameters: {sum(p.numel() for p in model.parameters()):,}")
        best_loss = train_model(model, tokenizer, self.device, self.config,
                                self.config.basemodel_save_path, self._logger())
        self._record_phase("predictor", best_loss, self.config.basemodel_best_model_path)
        del model, tokenizer
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return True

    def _record_phase(self, phase, loss, checkpoint):
        if self.rank != 0:
            return
        path = Path(self.config.base_save_path) / "training_manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else {}
        manifest.update(split_manifest(load_candles(self.config), self.config))
        manifest.update({"tokenizer_path": self.config.finetuned_tokenizer_path,
                         "pretrained_predictor": self.config.pretrained_predictor_path,
                         "from_scratch": self.config.from_scratch,
                         "lookback_window": self.config.lookback_window,
                         "predict_window": self.config.predict_window,
                         "seed": self.config.seed})
        manifest.setdefault("completed_phases", {})[phase] = {
            "best_validation_loss": float(loss), "checkpoint": checkpoint}
        path.write_text(json.dumps(manifest, indent=2) + "\n")

    def run_training(self):
        self._setup_distributed()
        for path in (self.config.tokenizer_save_path, self.config.basemodel_save_path):
            Path(path).mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        frame = load_candles(self.config)
        split_manifest(frame, self.config)
        try:
            if self.config.train_tokenizer:
                self.train_tokenizer_phase()
            if self.config.train_basemodel:
                self.train_basemodel_phase()
            print(f"Training completed in {(time.monotonic() - started) / 60:.2f} minutes")
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-basemodel", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    trainer = SequentialTrainer(args.config)
    if args.skip_tokenizer:
        trainer.config.train_tokenizer = False
    if args.skip_basemodel:
        trainer.config.train_basemodel = False
    if args.skip_existing:
        trainer.config.skip_existing = True
    try:
        success = trainer.run_training()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
