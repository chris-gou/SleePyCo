import os
import glob
import torch
import numpy as np
from transform import *
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.signal import resample_poly
import zipfile
 
 
def npz_shape(path, key='x'):
    """Shape of one array inside an .npz, read from the header only (no data load)."""
    with zipfile.ZipFile(path) as z, z.open(f'{key}.npy') as f:
        major, _ = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        shape, _, _ = read(f)
    return shape
 
DB_PATH = "/mnt/truenas_db/user/christina"

class EEGDataLoader(Dataset):

    def __init__(self, config, fold, set='train'):

        self.set = set
        self.fold = fold

        self.sr = 100        
        self.dset_cfg = config['dataset']
        
        self.root_dir = self.dset_cfg['root_dir']
        self.dset_name = self.dset_cfg['name']
        self.num_splits = self.dset_cfg['num_splits']
        self.eeg_channel = self.dset_cfg['eeg_channel']
        
        self.seq_len = self.dset_cfg['seq_len']
        self.target_idx = self.dset_cfg['target_idx']
        
        self.training_mode = config['training_params']['mode']
        self.fs = config['training_params']['fs']
        self.epoch_duration = config['training_params']['epoch_duration']
        window = int(self.epoch_duration * self.fs)
        self.hop = config['training_params']['hop']

        # fix datasets
        self.dataset_path = config['dataset']['dataset_path'] if config['dataset']['dataset_path'] else os.path.join(self.root_dir, 'dset', self.dset_name, 'npz')
        print(f"[INFO] Dataset path: {self.dataset_path}")
        self.inputs, self.labels, self.epochs = self.split_dataset()
        
        if self.epoch_duration != 30:
            # check first if the data is actually pre-split
            
            hop = int((self.hop if self.hop>0 else self.epoch_duration) * self.fs)
            if self.dset_name == 'SHHS':
                print(f"[INFO] Using lazy splitting for SHHS with epoch duration {self.epoch_duration}s and hop {hop} samples")
                self.inputs, self.labels, self.epochs = self.split_pre_epoched_lazy(self.inputs, self.labels, window, hop)
            else:
                self.inputs, self.labels, self.epochs = self.split_pre_epoched(self.inputs, self.labels, window, hop)

                # Verify the result
                expected = int(self.epoch_duration*self.sr)
                actual = self.inputs[0].shape[1]
                print(f"[INFO] Expected samples per epoch: {expected}, got: {actual}")
                print(f"[INFO] Total epochs after split: {len(self.epochs)}")
                assert actual == expected

        if self.training_mode == 'pretrain':
            self.transform = Compose(
                transforms=[
                    RandomAmplitudeScale(),
                    RandomTimeShift(),
                    RandomDCShift(),
                    RandomZeroMasking(),
                    RandomAdditiveGaussianNoise(),
                    RandomBandStopFilter(),
                ]
            )
            self.two_transform = TwoTransform(self.transform)
        
    def __len__(self):
        return len(self.epochs)

    def _to_target_sr(self, x, fs):
        if fs == self.sr:
           return x
        n = x.shape[0]
        flat = resample_poly(x.reshape(-1), self.sr, fs)
        return flat.reshape(n, -1).astype(np.float32)

    def __getitem__(self, idx):
        n_sample = int(self.epoch_duration) * self.sr * self.seq_len
        file_idx, start, seq_len = self.epochs[idx]
        # inputs = self.inputs[file_idx][idx:idx+seq_len]

        if self.epoch_duration != 30 and self.dset_name == 'SHHS':
            window = int(self.epoch_duration * self.sr)
            hop = int((self.hop if self.hop > 0 else self.epoch_duration) * self.fs)
            starts = list(range(0, int(30 * self.sr) - window + 1, hop))
            sub_per_base = len(starts)

            if not hasattr(self, '_mmap_cache'):
                self._mmap_cache = {}
            if file_idx not in self._mmap_cache:
                self._mmap_cache[file_idx] = np.load(self.inputs[file_idx], mmap_mode='r')
            npz_file = self._mmap_cache[file_idx]

            # npz_file = np.load(self.inputs[file_idx], mmap_mode='r')
            raw_inputs = []
            raw_labels = []
            for k in range(seq_len):
                flat = start + k
                bi = flat // sub_per_base
                si = flat % sub_per_base
                raw_inputs.append(npz_file['x'][bi, starts[si]:starts[si] + window])
                # raw_labels.append(npz_file['y'][flat])
                raw_labels.append(self.labels[file_idx][flat]) # pre-split labels

            inputs = np.stack(raw_inputs)  # (seq_len, window)
            labels = np.array(raw_labels)            
        else:
            if not hasattr(self, '_mmap_cache'):
                self._mmap_cache = {}
            data = self.inputs[file_idx]
            if isinstance(data, str):
                with np.load(self.inputs[file_idx]) as f:
                    inputs = self._to_target_sr(f['x'][start:start + seq_len], int(f['fs']))
                labels = self.labels[file_idx][start:start + seq_len]
                # if file_idx not in self._mmap_cache:
                #     self._mmap_cache[file_idx] = np.load(self.inputs[file_idx], mmap_mode='r')
                # npz_file = self._mmap_cache[file_idx]
                # inputs = npz_file['x'][start:start + seq_len]
                # inputs = self._to_target_sr(inputs, int(npz_file['fs']))
                # labels = npz_file['y'][start:start + seq_len]
            else:
                npz_file = data
                inputs = npz_file[start:start + seq_len]
                labels = self.labels[file_idx][start:start + seq_len]

        if self.set == 'train':
            if self.training_mode == 'pretrain':
                assert seq_len == 1
                input_a, input_b = self.two_transform(inputs)
                input_a = torch.from_numpy(input_a).float()
                input_b = torch.from_numpy(input_b).float()
                inputs = [input_a, input_b]
            elif self.training_mode in ['scratch', 'fullyfinetune', 'freezefinetune']:
                inputs = inputs.reshape(1, n_sample)
                inputs = torch.from_numpy(inputs).float()
            else:
                raise NotImplementedError
        else:
            if not self.training_mode == 'pretrain':
                inputs = inputs.reshape(1, n_sample)
            inputs = torch.from_numpy(inputs).float()
        
        labels = self.labels[file_idx][start:start+seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]
        
        return inputs, labels

    def get_subject(self, file_idx):
        """All epochs of one recording, preprocessed like __getitem__.
        Returns (n_epochs, samples_per_epoch) and labels. Used by streaming eval."""
        data = self.inputs[file_idx]
        labels = self.labels[file_idx]
        if not isinstance(data, str):
            return data, labels  # already in memory (non-SHHS)
 
        with np.load(data) as f:
            x = f['x']
            fs = int(f['fs'])
 
        if self.epoch_duration == 30:
            return self._to_target_sr(x, fs), labels
 
        # SHHS sub-epochs: same windows and order as the lazy branch of __getitem__
        window = int(self.epoch_duration * self.sr)
        hop = int((self.hop if self.hop > 0 else self.epoch_duration) * self.fs)
        starts = range(0, int(30 * self.sr) - window + 1, hop)
        x = np.stack([x[:, s:s + window] for s in starts], axis=1).reshape(-1, window)
        return x, labels
    
    def _check_no_overlap(self, fname_dict):
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            overlap = set(fname_dict[a]) & set(fname_dict[b])
            assert not overlap, (
                f"Fold {self.fold}: {len(overlap)} files in both {a} and {b}, "
                f"e.g. {sorted(overlap)[:5]}"
            )
        for mode, files in fname_dict.items():
            assert len(files) == len(set(files)), f"Fold {self.fold}: duplicate files in {mode}"
    
    def split_dataset(self):

        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.eeg_channel)
        print(f"[INFO] Loading data from {data_root}")
        data_fname_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(data_root, '*.npz')))]
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(self.dset_name)), allow_pickle=True)
        
        assert len(split_idx_list) == self.num_splits
        print(f"[INFO] Dataset {self.dset_name} has {len(data_fname_list)} files, split into {self.num_splits} folds")
        if self.dset_name == 'Sleep-EDF-2013':
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx == self.fold - 1:
                    data_fname_dict['test'].append(data_fname_list[i])
                elif subject_idx in split_idx_list[self.fold - 1]:
                    data_fname_dict['val'].append(data_fname_list[i])
                else:
                    data_fname_dict['train'].append(data_fname_list[i])    

        elif self.dset_name == 'Sleep-EDF-2018':
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])
                    
        elif self.dset_name == 'MASS' or self.dset_name == 'Physio2018' or self.dset_name == 'SHHS':
            for i in range(len(data_fname_list)):
                if i in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])
        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dataset))
        self._check_no_overlap(data_fname_dict)
        assert len(data_fname_dict[self.set]) > 0, f"Fold {self.fold}: no files for {self.set}"
        print(f"[INFO] Fold {self.fold}: {len(data_fname_dict['train'])} train, {len(data_fname_dict['val'])} val, {len(data_fname_dict['test'])} test files")
        
        for data_fname in tqdm(data_fname_dict[self.set]): # progress bar for loading files
            fpath = os.path.join(data_root, data_fname)

            if self.dset_name == 'SHHS':
                with np.load(fpath) as f:
                    labels.append(np.array(f['y']))
                inputs.append(fpath)
            else:
                # npz_file = np.load(fpath, mmap_mode='r')
                # inputs.append(npz_file['x'])
                # labels.append(npz_file['y'])
                npz_file = np.load(fpath)
                inputs.append(self._to_target_sr(npz_file['x'], int(npz_file['fs'])))
                labels.append(npz_file['y'])
                assert inputs[-1].shape[1] == 30 * self.sr, f"{data_fname}: {inputs[-1].shape}"
                
            seq_len = self.seq_len
            if self.dset_name== 'MASS' and ('-02-' in data_fname or '-04-' in data_fname or '-05-' in data_fname):
                seq_len = int(self.seq_len * 1.5)

            valid = (labels[-1] >= 0) & (labels[-1] < 5)

            for i in range(len(labels[-1]) - seq_len + 1):
                if valid[i:i + seq_len].all():
                    epochs.append([file_idx, i, seq_len])
            file_idx += 1
        return inputs, labels, epochs

    def split_pre_epoched(self, inputs, labels, window, hop, inherit=True):
        """
        Split epochs, epochs are list of per-file arrays
        """
        new_inputs, new_epochs, new_labels = [], [], []
        file_idx = 0

        for x_file, y_file in zip(inputs, labels):
            N, base_len = x_file.shape[0], x_file.shape[1]
            assert base_len >= window
            starts = list(range(0, base_len-window+1, hop))

            x_out, y_out = [], []
            sub_labels = []

            for i in range(N):
                for s in starts:
                    x_out.append(x_file[i, s:s+window])
                    y_out.append(y_file[i] if inherit else None)
                if len(x_out) < self.seq_len:
                    # print(f"[WARN] file {i} skipped: only {len(x_out)} sub-windows < seq_len {self.seq_len}")
                    continue

            x_out = np.array(x_out)
            y_out = np.array(y_out, dtype=y_file.dtype)

            new_inputs.append(x_out)
            new_labels.append(y_out)
            valid = (y_out >= 0) & (y_out < 5)
            for i in range(len(y_out)-self.seq_len+1):
                if valid[i:i+self.seq_len].all():
                    new_epochs.append([file_idx, i , self.seq_len])
            file_idx+=1
        
        return new_inputs, new_labels, new_epochs

    def split_pre_epoched_lazy(self, inputs, labels, window, hop): 
        new_epochs, new_labels_flat,new_inputs, file_idx = [], [], [], 0

        for x_file, y_file in tqdm(zip(inputs, labels)):
            if isinstance(x_file, str):
                N, base_len = npz_shape(x_file)[:2]
            else:
                N, base_len = x_file.shape[0], x_file.shape[1]
                
            assert base_len >= window
            starts = list(range(0, base_len - window + 1, hop))

            sub_labels = []
            for i in range(N):
                for s in starts:
                    sub_labels.append(y_file[i])

            if len(sub_labels) < self.seq_len:
                # print(f"[WARN] file {file_idx} skipped: only {len(sub_labels)} sub-windows")
                # file_idx += 1
                continue

            sub_labels = np.array(sub_labels, dtype=y_file.dtype)
            new_inputs.append(x_file)
            new_labels_flat.append(sub_labels)

            for i in range(len(sub_labels) - self.seq_len + 1):
                new_epochs.append([file_idx, i, self.seq_len])
            file_idx += 1

        return new_inputs, new_labels_flat, new_epochs