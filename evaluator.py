import threading
import queue
import time
from collections import deque

import numpy as np
import torch
import pynvml

from utils import progress_bar


class StreamingEvaluator:
    """
    Producer/consumer FIFO simulating live epoch-by-epoch arrival.

    Producer: pushes raw epochs one at a time, subject by subject.
    Queue size 1: at most one pending epoch.
    Consumer: keeps a context buffer deque(maxlen=seq_len), reset at
    every new subject. Once full, each new epoch triggers one forward
    pass (batch size 1); the prediction for the newest epoch
    (target_idx = -1) is kept. The first seq_len-1 epochs of each
    subject get no prediction (cold start).

    We don't sleep for the real epoch duration (e.g. 30 s) between
    epochs: only processing time counts toward the latency budget.
    Timed per push: window stacking + host-to-device copy + forward pass.
    Works on CPU and GPU (GPU sync only when device is cuda).
    """

    def __init__(self, model, device, gpu_handle=None, seq_len=None, max_preds=None):
        assert seq_len is not None, "seq_len (FIFO capacity L) is required"
        self.model = model
        self.device = device
        self.seq_len = seq_len
        self.max_preds = max_preds  # stop after this many predictions (None = full test set)
        self.queue = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.results = []
        self.gpu_handle = gpu_handle  # pynvml handle, or None (CPU) to skip GPU sampling
        print(f"[INFO] StreamingEvaluator initialized | device: {self.device} | L = {self.seq_len} "
              f"| GPU sampling: {self.gpu_handle is not None}")

    def _producer(self, dataset):
        for file_idx in range(len(dataset.labels)):
            # loads one recording at a time (SHHS stores file paths, not arrays);
            # runs in the producer thread, outside the timed region
            subj_inputs, subj_labels = dataset.get_subject(file_idx)
            new_subject = True
            for epoch, label in zip(subj_inputs, subj_labels):
                if self.stop.is_set():
                    break  # consumer reached max_preds
                self.queue.put((epoch, label, new_subject))
                new_subject = False
            if self.stop.is_set():
                break
        self.queue.put(None)  # sentinel: stream ended

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    @torch.no_grad()
    def _consumer(self, total):
        self.model.eval()
        buffer = deque(maxlen=self.seq_len)
        n = 0
        t_start = time.time()

        while True:
            item = self.queue.get()
            if item is None:
                break
            epoch, label, new_subject = item

            if new_subject:
                buffer.clear()  # no context leaks across subjects
            buffer.append(epoch)
            if len(buffer) < self.seq_len:
                continue  # cold start: window not full yet

            self._sync()
            t0 = time.perf_counter()

            window = np.stack(buffer).reshape(1, 1, -1)  # (1, 1, L * samples)
            x = torch.from_numpy(window).float().to(self.device)
            outputs = self.model(x)

            self._sync()
            latency_ms = (time.perf_counter() - t0) * 1000

            # sum multi-scale outputs, same as the non-streaming path
            if isinstance(outputs, (list, tuple)):
                outputs = torch.stack(outputs, dim=0).sum(dim=0)
            output_sum = outputs[-1:].cpu()  # newest epoch only, shape (1, C)

            gpu_util, gpu_mem_mb = None, None
            if self.gpu_handle is not None:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_util, gpu_mem_mb = util.gpu, mem.used / (1024 ** 2)

            self.results.append({
                'latency_ms': latency_ms,
                'gpu_util': gpu_util,
                'gpu_mem_mb': gpu_mem_mb,
                'label': torch.tensor([int(label)]),
                'output_sum': output_sum,
            })
            progress_bar(n, total, f'Streaming eval | latency: {latency_ms:.1f} ms')
            n += 1

            if self.max_preds is not None and n >= self.max_preds:
                self.stop.set()
                while self.queue.get() is not None:  # drain until the producer's sentinel
                    pass
                break

        print(f"\n[TIMING] consumer processed {n} predictions in {time.time() - t_start:.1f}s total")

    def run(self, dataset):
        # number of predictions = epochs minus cold start, per subject
        total = sum(max(len(lbl) - self.seq_len + 1, 0) for lbl in dataset.labels)
        if self.max_preds is not None:
            total = min(total, self.max_preds)

        producer_thread = threading.Thread(target=self._producer, args=(dataset,))
        producer_thread.start()
        self._consumer(total)
        producer_thread.join()
        return self.results