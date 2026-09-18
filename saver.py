from datetime import datetime
import os
import glob
import json
import shutil
import mlx.core as mx
from mlx.utils import tree_flatten
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PhobetorCheckpointManager:
    def __init__(self, checkpoint_dir="./checkpoints", max_checkpoints=15):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.history_file = os.path.join(checkpoint_dir, "history.json")
        self.best_dir = os.path.join(checkpoint_dir, "best_model")
        self.best_loss = float("inf")

        self.history = {
            "steps": [],
            "losses": [],
            "lrs": [],
            "tok_per_sec": []
        }
        self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    self.history = data.get("history", self.history)
                    self.best_loss = data.get("best_loss", float("inf"))
                print(f"Loaded history: {len(self.history['steps'])} records. Best loss: {self.best_loss:.4f}")
            except Exception as e:
                print(f"Failed to load history: {e}")

    def log_step(self, step, loss, lr, tok_per_sec):
        self.history["steps"].append(int(step))
        self.history["losses"].append(float(loss))
        self.history["lrs"].append(float(lr))
        self.history["tok_per_sec"].append(float(tok_per_sec))

    def _generate_plot(self, save_path):
        if len(self.history["steps"]) < 2:
            return

        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150, sharex=True)

        steps = self.history["steps"]
        losses = self.history["losses"]
        lrs = self.history["lrs"]

        ax1.plot(steps, losses, color="#00ffcc", linewidth=1.5, label="Cross-Entropy Loss")
        if len(losses) >= 10:
            window = min(50, len(losses))
            smoothed = [
                sum(losses[max(0, i - window):i + 1]) / len(losses[max(0, i - window):i + 1])
                for i in range(len(losses))
            ]
            ax1.plot(steps, smoothed, color="#ffffff", linewidth=2.0, linestyle="--", label=f"Smoothed (MA-{window})")

        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Phobetor Training Dynamics", fontsize=14, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.4)
        ax1.legend(loc="upper right")

        ax2.plot(steps, lrs, color="#ff7700", linewidth=1.5, label="Learning Rate")
        ax2.set_ylabel("Learning Rate", fontsize=12)
        ax2.set_xlabel("Steps", fontsize=12)
        ax2.grid(True, linestyle=":", alpha=0.4)
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def save(self, model, step, loss, lr, total_tokens, elapsed_sec, sample_text=None):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        time_display = now.strftime("%Y-%m-%d %H:%M:%S")

        if not self.history["steps"] or self.history["steps"][-1] != int(step):
            avg_tok = self.history["tok_per_sec"][-1] if self.history["tok_per_sec"] else 0.0
            self.log_step(step, loss, lr, avg_tok)

        folder_name = f"ckpt_step_{step:07d}_loss_{loss:.4f}_{timestamp}"
        ckpt_folder = os.path.join(self.checkpoint_dir, folder_name)
        os.makedirs(ckpt_folder, exist_ok=True)

        weights_path = os.path.join(ckpt_folder, "model.safetensors")
        weights = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(weights_path, weights)

        metrics = {
            "timestamp": time_display,
            "step": int(step),
            "loss": float(loss),
            "lr": float(lr),
            "total_tokens_seen": int(total_tokens),
            "elapsed_time_hours": round(elapsed_sec / 3600, 2),
            "avg_tokens_per_sec": round(self.history["tok_per_sec"][-1] if self.history["tok_per_sec"] else 0.0, 1),
            "best_loss_so_far": float(min(self.best_loss, loss))
        }
        with open(os.path.join(ckpt_folder, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        plot_path = os.path.join(ckpt_folder, "training_plot.png")
        self._generate_plot(plot_path)

        if sample_text:
            with open(os.path.join(ckpt_folder, "sample_generation.txt"), "w", encoding="utf-8") as f:
                f.write(sample_text)

        with open(self.history_file, "w") as f:
            json.dump({"best_loss": min(self.best_loss, loss), "history": self.history}, f)

        print(f"\nCheckpoint saved: {ckpt_folder} | {time_display}")

        if loss < self.best_loss:
            self.best_loss = loss
            os.makedirs(self.best_dir, exist_ok=True)
            mx.save_safetensors(os.path.join(self.best_dir, "model.safetensors"), weights)
            with open(os.path.join(self.best_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            if os.path.exists(plot_path):
                shutil.copyfile(plot_path, os.path.join(self.best_dir, "training_plot.png"))
            print(f"New best record ({loss:.4f}) copied to {self.best_dir} | {time_display}")

        self._cleanup_old_folders()

    def _cleanup_old_folders(self):
        all_dirs = [
            d for d in glob.glob(os.path.join(self.checkpoint_dir, "ckpt_step_*"))
            if os.path.isdir(d)
        ]
        all_dirs.sort(key=os.path.getmtime)

        while len(all_dirs) > self.max_checkpoints:
            oldest_dir = all_dirs.pop(0)
            try:
                shutil.rmtree(oldest_dir)
                print(f"Removed old checkpoint folder {os.path.basename(oldest_dir)}")
            except OSError as e:
                print(f"Failed to remove {oldest_dir}: {e}")