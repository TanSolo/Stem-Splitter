import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# IMPORTANT: your file is datasets.py, so import from datasets (not dataset)
from dataset import MUSDBDataset, STEMS
from model import UNetMask


def stft_mag(x, n_fft=1024, hop=256):
    """
    x: [B, 2, T]
    returns magnitude: [B, 2, F, TT]
    """
    B, C, T = x.shape
    window = torch.hann_window(n_fft, device=x.device)
    mags = []
    for ch in range(C):
        X = torch.stft(
            x[:, ch],
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            return_complex=True,
            center=True,
        )  # [B, F, TT]
        mags.append(X.abs())
    return torch.stack(mags, dim=1)  # [B, 2, F, TT]


def main():
    # IMPORTANT: set this to your MUSDB-HQ folder
    musdb_root = r"D:\Datasets\musdbhq"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on device:", device)

    os.makedirs("checkpoints", exist_ok=True)

    # settings (clarity run)
    sr = 44100
    segment_seconds = 6.0
    batch_size = 2
    epochs = 50
    lr = 1e-4

    ds = MUSDBDataset(root=musdb_root, segment_seconds=segment_seconds, sr=sr)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
    )

    model = UNetMask(base=32, stems=len(STEMS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # resume support (stop anytime and continue later)
    resume_path = "checkpoints/last.pt"
    start_ep = 1
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_ep = ckpt["epoch"] + 1
        print(f"Resumed from {resume_path} at epoch {ckpt['epoch']}")

    for ep in range(start_ep, epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {ep}/{epochs}")

        running = 0.0
        steps = 0

        for mix, target in pbar:
            mix = mix.to(device, non_blocking=True)           # [B, 2, T]
            target = target.to(device, non_blocking=True)     # [B, 4, 2, T]

            mix_mag = stft_mag(mix)  # [B, 2, F, TT]

            tgt_mag = stft_mag(target.view(-1, 2, target.size(-1)))  # [B*4, 2, F, TT]
            tgt_mag = tgt_mag.view(
                target.size(0),
                len(STEMS),
                2,
                tgt_mag.size(-2),
                tgt_mag.size(-1),
            )  # [B,4,2,F,TT]

            pred_masks = model(mix_mag)  # [B,4,2,Fp,TTp]

            # crop everything to common (F,TT) so shapes always match
            Fm, TTm = mix_mag.shape[-2], mix_mag.shape[-1]
            Fp, TTp = pred_masks.shape[-2], pred_masks.shape[-1]
            Ft = min(Fm, Fp)
            TTt = min(TTm, TTp)

            mix_mag_c = mix_mag[:, :, :Ft, :TTt]
            pred_masks_c = pred_masks[:, :, :, :Ft, :TTt]
            tgt_mag_c = tgt_mag[:, :, :, :Ft, :TTt]

            pred_mag = pred_masks_c * mix_mag_c.unsqueeze(1)  # [B,4,2,F,TT]
            loss = F.l1_loss(pred_mag, tgt_mag_c)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=running / steps)

        # epoch checkpoint (weights only)
        ckpt_path = f"checkpoints/unetmask_ep{ep}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print("Saved:", ckpt_path)

        # resume checkpoint (weights + optimizer + epoch)
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep},
            "checkpoints/last.pt",
        )
        print("Saved resume checkpoint: checkpoints/last.pt")


if __name__ == "__main__":
    main()
