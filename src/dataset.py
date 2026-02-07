import os
import random
import torch
import torchaudio
import numpy as np
import soundfile as sf

STEMS = ["vocals", "drums", "bass", "other"]

class MUSDBDataset(torch.utils.data.Dataset):
    def __init__(self, root, segment_seconds=6.0, sr=44100):
        self.root = root
        self.sr = sr
        self.segment_samples = int(segment_seconds * sr)

        self.tracks = []
        for split in ["train"]:
            split_dir = os.path.join(root, split)
            for track in os.listdir(split_dir):
                self.tracks.append(os.path.join(split_dir, track))

    def __len__(self):
        return 1000  # fake "epoch size"

    def _load_resample(self, path):
        wav, sr = sf.read(path, always_2d=True)  # [T, C]
        wav = torch.from_numpy(wav).float().T  # -> [C, T]

        # ensure stereo
        if wav.size(0) == 1:
            wav = wav.repeat(2, 1)
        elif wav.size(0) > 2:
            wav = wav[:2]

        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        return wav

    def __getitem__(self, idx):
        track_dir = random.choice(self.tracks)

        mix = self._load_resample(os.path.join(track_dir, "mixture.wav"))
        stems = [self._load_resample(os.path.join(track_dir, f"{s}.wav")) for s in STEMS]

        T = mix.size(1)
        if T > self.segment_samples:
            start = random.randint(0, T - self.segment_samples)
            mix = mix[:, start:start+self.segment_samples]
            stems = [s[:, start:start+self.segment_samples] for s in stems]

        target = torch.stack(stems, dim=0)  # [4, 2, T]
        return mix, target
