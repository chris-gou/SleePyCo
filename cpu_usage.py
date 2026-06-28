import psutil
import threading
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import json
class CPUSampler:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples = []
        self.timestamps = []
        self._stop = threading.Event()

    def start(self):
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[INFO] CPU sampler started with interval: {self.interval} seconds")

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(interval=None))
            self.timestamps.append(time.time() - self._t0)
            time.sleep(self.interval)

    def stats(self):
        if not self.samples:
            return {}
        return {
            'mean': float(np.mean(self.samples)),
            'p95':  float(np.percentile(self.samples, 95)),
            'max':  float(np.max(self.samples)),
        }
    
def plot_cpu_usage_per_fold(cpu_samplers, config_name, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(len(cpu_samplers), 1,
                             figsize=(12, 3 * len(cpu_samplers)),
                             sharex=False)
    if len(cpu_samplers) == 1:
        axes = [axes]

    for fold_idx, sampler in enumerate(cpu_samplers):
        times   = np.array(sampler.timestamps)
        samples = np.array(sampler.samples)
        stats   = sampler.stats()

        ax = axes[fold_idx]
        ax.plot(times, samples, linewidth=0.8, color='steelblue')
        ax.axhline(stats['mean'], color='red',    linestyle='--', linewidth=1, label=f"mean={stats['mean']:.1f}%")
        ax.axhline(stats['p95'],  color='orange', linestyle=':',  linewidth=1, label=f"p95={stats['p95']:.1f}%")
        ax.axhline(stats['max'],  color='black',  linestyle=':',  linewidth=1, label=f"max={stats['max']:.1f}%")
        ax.set_ylabel('CPU %')
        ax.set_ylim(0, 100)
        ax.set_title(f'Fold {fold_idx + 1}')
        ax.set_xlabel('Time (s)')
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'CPU Usage During Streaming Evaluation\n{config_name}', fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f'{config_name}_cpu_per_fold.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'[INFO] CPU plot saved to {out_path}')

if __name__ == "__main__":
    config_name = "configs/ablations_"