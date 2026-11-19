"""Robust physical distance, shrinkage Mahalanobis and convolutional autoencoder."""
import copy
import os

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn


def physical_scores(train, observations):
    center = np.median(train, axis=0)
    spread = np.subtract(*np.percentile(train, [75, 25], axis=0))
    return np.abs((observations - center) / np.maximum(spread, 1e-10)).mean(axis=1)


def mahalanobis_scores(train, observations):
    scaler = StandardScaler().fit(train)
    distribution = LedoitWolf().fit(scaler.transform(train))
    return np.sqrt(np.maximum(distribution.mahalanobis(scaler.transform(observations)), 0))


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = []
        for inputs, outputs in ((1, 16), (16, 32), (32, 64)):
            encoder.extend([nn.Conv2d(inputs, outputs, 3, 2, 1),
                            nn.BatchNorm2d(outputs), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder, nn.Conv2d(64, 8, 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 64, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=(1, 0)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=(1, 1)),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=(1, 1)),
        )

    def forward(self, x):
        reconstruction = self.decoder(self.encoder(x))
        return reconstruction[..., :x.shape[-2], :x.shape[-1]]


def check_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability()
    architecture = f"sm_{capability[0]}{capability[1]}"
    if architecture not in torch.cuda.get_arch_list():
        raise RuntimeError(f"PyTorch wheel lacks {architecture}: {torch.cuda.get_arch_list()}")
    sample = torch.ones((2, 1, 64, 100), device=device)
    layer = ConvAE().to(device)
    layer(sample).square().mean().backward()
    torch.cuda.synchronize()
    return device


def fit_autoencoder(features, train_ids, calib_ids, seed, config, device, checkpoint):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(config["cpu_threads"])
    train = features[train_ids]
    center = train.mean(axis=(0, 2), keepdims=True)
    spread = np.maximum(train.std(axis=(0, 2), keepdims=True), 1e-6)
    normalized = torch.from_numpy(((features - center) / spread)[:, None]).to(device)
    del train
    model = ConvAE().to(device)
    batch_size = config["batch_size"]

    @torch.inference_mode()
    def score(inputs):
        model.eval()
        return torch.cat([
            (model(batch) - batch).square().mean(dim=(1, 2, 3))
            for batch in inputs.split(batch_size)
        ])

    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        history = state["history"]
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                      weight_decay=config["weight_decay"], fused=device.type == "cuda")
        train = normalized[train_ids]
        calib = normalized[calib_ids]
        best_loss, best_state, best_epoch = float("inf"), None, 0
        history = []
        for epoch in range(1, config["max_epochs"] + 1):
            model.train()
            order = torch.randperm(len(train), device=device)
            for indices in order.split(batch_size):
                batch = train[indices]
                optimizer.zero_grad(set_to_none=True)
                (model(batch) - batch).square().mean().backward()
                optimizer.step()
            loss = score(calib).mean().item()
            if not np.isfinite(loss):
                raise ValueError("Non-finite AE calibration loss")
            history.append(loss)
            print(f"{checkpoint.stem} epoch {epoch:02d} calibration MSE {loss:.8g}", flush=True)
            if loss < best_loss:
                best_loss, best_state, best_epoch = loss, copy.deepcopy(model.state_dict()), epoch
            if epoch - best_epoch >= config["patience"]:
                break
        model.load_state_dict(best_state)
        checkpoint.parent.mkdir(exist_ok=True, parents=True)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"model": best_state, "history": history}, temporary)
        temporary.replace(checkpoint)
    return score(normalized).cpu().numpy(), history
