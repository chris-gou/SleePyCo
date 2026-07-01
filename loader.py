import os
import glob
import torch
import numpy as np
from transform import *
from torch.utils.data import Dataset
from tqdm import tqdm

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
        self.dataset_path = os.path.join(self.root_dir, 'dset', self.dset_name, 'npz')
        self.inputs, self.labels, self.epochs = self.split_dataset()
        
        if self.epoch_duration != 30:
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

    def __getitem__(self, idx):
        n_sample = int(self.epoch_duration) * self.sr * self.seq_len
        file_idx, start, seq_len = self.epochs[idx]
        inputs = self.inputs[file_idx][idx:idx+seq_len]

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
            if file_idx not in self._mmap_cache:
                self._mmap_cache[file_idx] = np.load(self.inputs[file_idx], mmap_mode='r')
            npz_file = self._mmap_cache[file_idx]
            inputs = npz_file['x'][start:start + seq_len]
            labels = npz_file['y'][start:start + seq_len]

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
        
        labels = self.labels[file_idx][idx:idx+seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]
        
        return inputs, labels

    def split_dataset(self):

        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.eeg_channel)
        data_fname_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(data_root, '*.npz')))]
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(self.dset_name)), allow_pickle=True)

        assert len(split_idx_list) == self.num_splits
    
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
            
        for data_fname in tqdm(data_fname_dict[self.set]): # progress bar for loading files
            fpath = os.path.join(data_root, data_fname)

            if self.dset_name == 'SHHS':
                npz_file = np.load(fpath, mmap_mode='r')
                labels.append(np.array(npz_file['y']))  # labels are tiny, safe to materialize
                inputs.append(fpath)                     # store path instead of data
                npz_file._mmap.close() if hasattr(npz_file, '_mmap') else None
            else:
                npz_file = np.load(fpath, mmap_mode='r')
                inputs.append(npz_file['x'])
                labels.append(npz_file['y'])

            seq_len = self.seq_len
            if self.dset_name== 'MASS' and ('-02-' in data_fname or '-04-' in data_fname or '-05-' in data_fname):
                seq_len = int(self.seq_len * 1.5)
            for i in range(len(npz_file['y']) - seq_len + 1):
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

            for i in range(len(y_out)-self.seq_len+1):
                new_epochs.append([file_idx, i , self.seq_len])
            file_idx+=1
        
        return new_inputs, new_labels, new_epochs

    def split_pre_epoched_lazy(self, inputs, labels, window, hop): 
        new_epochs, new_labels_flat,new_inputs, file_idx = [], [], [], 0

        for x_file, y_file in zip(inputs, labels):
            if isinstance(x_file, str):
                npz_file = np.load(x_file, mmap_mode='r')
                N, base_len = npz_file['x'].shape[0], npz_file['x'].shape[1]
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