import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import SafetensorError, safe_open


LEGACY_VALIDATION_CONFIG = {
    "shuffle_seed": 1338,
    "mask_seed": 20_000_000,
    "batches": 64,
    "batch_size": 2,
    "seq_len": 1024,
    "mask_eps": 1e-3,
}


def require_finite(tree, label):
    leaves = [value for _, value in tree_flatten(tree)]
    if leaves and not bool(mx.all(mx.stack([mx.all(mx.isfinite(v)) for v in leaves])).item()):
        raise FloatingPointError(f"Non-finite {label}")


def optimizer_steps(state):
    return [int(s["step"].item()) for s in state.get("states", [state])]


@dataclass
class TrainingProgress:
    step: int = 0
    tokens_seen: int = 0
    train_loss: float | None = None
    safe_to_save: bool = True

    def update(self, model, optimizer, grads, loss, tokens_per_step, max_grad_norm):
        if not self.safe_to_save:
            raise RuntimeError("Previous optimizer update did not finish; reload a checkpoint")
        if not math.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss}")

        grads, grad_norm = opt.clip_grad_norm(grads, max_grad_norm)
        norm_value = float(grad_norm.item())
        if not math.isfinite(norm_value):
            raise FloatingPointError(f"Non-finite gradient norm: {norm_value}")

        self.safe_to_save = False
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        require_finite(model.parameters(), "updated model weights")
        require_finite(optimizer.state, "updated optimizer state")
        next_step = self.step + 1
        if any(step != next_step for step in optimizer_steps(optimizer.state)):
            raise RuntimeError("Optimizer counters disagree with the completed update")

        self.step = next_step
        self.tokens_seen += tokens_per_step
        self.train_loss = loss
        self.safe_to_save = True
        return norm_value


def checkpoint_step(path):
    path = Path(path)
    if path.name == "emergency_checkpoint":
        try:
            step = json.loads((path / "metrics.json").read_text())["step"]
            return step if type(step) is int and step >= 0 else -1
        except (OSError, ValueError, KeyError, TypeError):
            return -1
    match = re.fullmatch(r"ckpt_step_(\d+)(?:_\d+)?", Path(path).name)
    return int(match.group(1)) if match else -1


def checkpoint_paths(directory, include_emergency=True):
    root = Path(directory)
    paths = [p for p in root.glob("ckpt_step_*") if p.is_dir() and checkpoint_step(p) >= 0]
    if include_emergency:
        emergency = _recover_slot(root, "emergency_checkpoint")
        if emergency.is_dir():
            paths.append(emergency)
    return sorted(
        paths,
        key=lambda p: (checkpoint_step(p), p.stat().st_mtime_ns),
        reverse=True,
    )


def inspect_checkpoint(path):
    path = Path(path)
    with (path / "metrics.json").open() as f:
        metrics = json.load(f)
    step = metrics["step"]
    if not isinstance(step, int) or step < 0 or step != checkpoint_step(path):
        raise ValueError("Checkpoint directory and metadata step disagree")
    if not isinstance(metrics.get("tokens_seen"), int) or metrics["tokens_seen"] < 0:
        raise ValueError("Invalid tokens_seen")
    if not math.isfinite(metrics["train_loss"]):
        raise ValueError("Non-finite checkpoint training loss")
    with safe_open(str(path / "model.safetensors"), framework="numpy") as f:
        if not f.keys():
            raise ValueError("Empty model checkpoint")
    with safe_open(str(path / "optimizer.safetensors"), framework="numpy") as f:
        keys = [k for k in f.keys() if k == "step" or re.fullmatch(r"states\.\d+\.step", k)]
        steps = [int(f.get_tensor(k).item()) for k in keys]
    if not steps or any(s != step for s in steps):
        raise ValueError(f"metadata step={step}, optimizer steps={steps}; update may be incomplete")
    return metrics


def restore_checkpoint(path, model, optimizer, model_config, vocab_size):
    expected_config = json.loads(json.dumps(model_config))
    path = Path(path)
    metrics = inspect_checkpoint(path)
    if metrics.get("model_config") != expected_config or metrics.get("vocab_size") != vocab_size:
        raise ValueError("Model configuration or vocabulary does not match")
    model.load_weights(str(path / "model.safetensors"))
    optimizer.state = tree_unflatten(mx.load(str(path / "optimizer.safetensors")))
    mx.eval(model.parameters(), optimizer.state)
    require_finite(model.parameters(), "checkpoint weights")
    require_finite(optimizer.state, "checkpoint optimizer state")
    print(f"Resuming complete checkpoint: {path}")
    return TrainingProgress(metrics["step"], metrics["tokens_seen"], metrics["train_loss"]), metrics


def restore_latest(directory, model, optimizer, model_config, vocab_size):
    paths = checkpoint_paths(directory)
    for path in paths:
        try:
            return restore_checkpoint(path, model, optimizer, model_config, vocab_size)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, FloatingPointError, SafetensorError) as exc:
            print(f"Skipping checkpoint {path.name}: {exc}")
            continue
    if paths:
        raise RuntimeError("No valid resumable checkpoint found; existing files were preserved")
    return TrainingProgress(), {}


def validation_matches(metrics, config):
    return metrics.get("validation_config", LEGACY_VALIDATION_CONFIG) == config


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_snapshot(path, model, metrics, optimizer=None):
    model.save_weights(str(path / "model.safetensors"))
    if optimizer is not None:
        mx.save_safetensors(str(path / "optimizer.safetensors"), tree_flatten(optimizer.state, destination={}))
    with (path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)
    for file in path.iterdir():
        with file.open("rb") as f:
            os.fsync(f.fileno())
    _sync_directory(path)


def _recover_slot(root, name):
    path, previous = root / name, root / f".{name}.previous"
    if not path.exists() and previous.exists():
        os.replace(previous, path)
        _sync_directory(root)
    return path


def _publish_slot(root, temp, name):
    path = _recover_slot(root, name)
    previous = root / f".{name}.previous"
    if previous.exists():
        shutil.rmtree(previous)
    if path.exists():
        os.replace(path, previous)
        _sync_directory(root)
    try:
        os.replace(temp, path)
        _sync_directory(root)
    except BaseException:
        _recover_slot(root, name)
        raise
    if previous.exists():
        shutil.rmtree(previous)
        _sync_directory(root)
    return path


def save_checkpoint(directory, model, optimizer, progress, metadata, max_checkpoints=3, kind="periodic"):
    if kind not in {"periodic", "emergency"}:
        raise ValueError("Unknown checkpoint kind")
    if not progress.safe_to_save:
        raise RuntimeError("Refusing to save an interrupted or invalid optimizer update")
    if max_checkpoints < 1:
        raise ValueError("max_checkpoints must be positive")
    if any(step != progress.step for step in optimizer_steps(optimizer.state)):
        raise RuntimeError("Refusing to save mismatched optimizer counters")
    require_finite(model.parameters(), "checkpoint weights")
    require_finite(optimizer.state, "checkpoint optimizer state")
    metrics = {
        **metadata, "checkpoint_version": 2, "step": progress.step,
        "checkpoint_kind": kind,
        "train_loss": progress.train_loss, "tokens_seen": progress.tokens_seen,
    }
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"ckpt_step_{progress.step:07d}"
    if path.exists():
        path = root / f"ckpt_step_{progress.step:07d}_{time.time_ns()}"
    temp = Path(tempfile.mkdtemp(prefix=".tmp_ckpt_", dir=root))
    try:
        _write_snapshot(temp, model, metrics, optimizer)
        if kind == "emergency":
            path = _publish_slot(root, temp, "emergency_checkpoint")
        else:
            os.replace(temp, path)
            _sync_directory(root)
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
    if kind == "emergency":
        print(f"Emergency checkpoint saved: {path}")
        return path
    valid = []
    for old in checkpoint_paths(root, include_emergency=False):
        try:
            inspect_checkpoint(old)
        except (ValueError, KeyError, TypeError, OSError, SafetensorError):
            continue
        valid.append(old)
    for old in valid[max_checkpoints:]:
        shutil.rmtree(old, ignore_errors=True)
    print(f"Checkpoint saved: {path}")
    return path


def recover_best(directory):
    root = Path(directory)
    path, previous = root / "best_model", root / ".best_model.previous"
    if not path.exists() and previous.exists():
        os.replace(previous, path)
        _sync_directory(root)
    return path


def load_best_loss(directory, validation_config, metric="val_loss"):
    path = recover_best(directory) / "metrics.json"
    if not path.exists():
        return float("inf")
    try:
        with path.open() as f:
            metrics = json.load(f)
        if not validation_matches(metrics, validation_config):
            print("Best validation score uses a different evaluation setup; starting a new comparison")
            return float("inf")
        value = float(metrics["validation_metrics"][metric])
        return value if math.isfinite(value) else float("inf")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Could not read previous best score: {exc}")
        return float("inf")


def save_best(directory, model, metrics):
    require_finite(model.parameters(), "best model weights")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = recover_best(root)
    previous = root / ".best_model.previous"
    temp = Path(tempfile.mkdtemp(prefix=".tmp_best_", dir=root))
    try:
        _write_snapshot(temp, model, metrics)
        if previous.exists():
            shutil.rmtree(previous)
        if path.exists():
            os.replace(path, previous)
            _sync_directory(root)
        try:
            os.replace(temp, path)
            _sync_directory(root)
        except BaseException:
            recover_best(root)
            raise
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
    if previous.exists():
        shutil.rmtree(previous, ignore_errors=True)
