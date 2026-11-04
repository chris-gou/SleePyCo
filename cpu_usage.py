import psutil
import threading
import time
import numpy as np
import os
import pynvml
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import threading

class EventMarker:
    def __init__(self, start_time=None):
        self.start_time = start_time or time.time()
        self.events = []  # list of (elapsed_seconds, label)

    def mark(self, label):
        elapsed = time.time() - self.start_time
        self.events.append((elapsed, label))
        print(f"[EVENT] {label} @ {elapsed:.1f}s")
class PeriodicPlotter:
    def __init__(self, cpu_sampler, gpu_sampler, out_path, interval_sec=2 * 3600):
        self.cpu_sampler = cpu_sampler
        self.gpu_sampler = gpu_sampler
        self.out_path = out_path
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop_event.wait(self.interval_sec):
            plot_resource_usage(self.cpu_sampler, self.gpu_sampler, self.out_path)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

class GPUSampler:
    def __init__(self, interval=0.05):
        self.interval = interval
        self._util_samples = []
        self._mem_samples = []
        self._timestamps = []
        self._stop = threading.Event()
        self._thread = None
    
    def _run(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # single GPU
        start_time = time.time()
        while not self._stop.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self._util_samples.append(util)
            self._mem_samples.append(mem.used / (1024 ** 2))
            self._timestamps.append(time.time() - start_time)
            time.sleep(self.interval)
        pynvml.nvmlShutdown()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[INFO] GPU sampler started with interval: {self.interval} seconds")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def stats(self):
        if not self._util_samples:
            return {'mean': None, 'p95': None, 'max': None,
                    'mem_mean': None, 'mem_max': None}
        import numpy as np
        return {
            'mean': float(np.mean(self._util_samples)),
            'p95': float(np.percentile(self._util_samples, 95)),
            'max': float(np.max(self._util_samples)),
            'mem_mean': float(np.mean(self._mem_samples)),
            'mem_max': float(np.max(self._mem_samples)),
        }

    def history(self):
        """Raw traces for plotting."""
        return {
            'timestamps': self._timestamps,
            'util': self._util_samples,
            'mem_mb': self._mem_samples,
        }

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

    def history(self):
        """Raw traces for plotting."""
        return {
            'timestamps': self.timestamps,
            'util': self.samples,
            'mem_mb': [0] * len(self.timestamps),  # Placeholder - adjust as needed
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

def plot_resource_usage(cpu_sampler, gpu_sampler, out_path, events=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    if cpu_sampler is not None:
        cpu_hist = cpu_sampler.history()  # adjust if your CPUSampler doesn't expose this yet
        axes[0].plot(cpu_hist['timestamps'], cpu_hist['util'])
        axes[0].set_ylabel('CPU (%)')

    if gpu_sampler is not None:
        gpu_hist = gpu_sampler.history()
        axes[1].plot(gpu_hist['timestamps'], gpu_hist['util'])
        axes[1].set_ylabel('GPU util (%)')

        axes[2].plot(gpu_hist['timestamps'], gpu_hist['mem_mb'])
        axes[2].set_ylabel('GPU mem (MB)')

    if events:
        for ax in axes:
            for t, label in events:
                ax.axvline(t, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        # label only on top subplot to avoid clutter
        for t, label in events:
            axes[0].text(t, axes[0].get_ylim()[1], label,
                        rotation=90, va='top', ha='right', fontsize=7, color='red')


    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    config_name = "configs/ablations_"


