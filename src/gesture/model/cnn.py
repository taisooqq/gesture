"""손 좌표 시계열을 1D CNN으로 학습하고 불러옵니다."""

import numpy as np


class SequenceClassifier:
    """프레임 순서를 보고 동작 점수를 냅니다. 랜덤포레스트와 같이 predict_proba를 제공합니다."""

    def __init__(self, network, tags):
        self.network = network
        self.classes_ = np.arange(len(tags))

    def predict_proba(self, sequence):
        import torch

        self.network.eval()
        tensor = torch.from_numpy(np.asarray(sequence, dtype=np.float32))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            logits = self.network(tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba[0]


def train_sequence_model(sequences, labels, class_count):
    """좌표 시계열로 작은 1D CNN을 맞춥니다."""
    import torch
    from torch import nn

    features = int(sequences.shape[-1])
    network = _Network(features, class_count)
    device = _device()
    network.to(device)
    counts = np.bincount(labels, minlength=class_count).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    xs = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
    ys = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    batch = 128
    network.train()
    for _epoch in range(20):
        order = torch.randperm(xs.shape[0])
        for start in range(0, xs.shape[0], batch):
            chosen = order[start : start + batch]
            optimizer.zero_grad()
            logits = network(xs[chosen].to(device))
            loss = loss_fn(logits, ys[chosen].to(device))
            loss.backward()
            optimizer.step()
    network.to("cpu")
    network.eval()
    return network


def save_sequence_model(path, network, tags, thresholds):
    import torch

    torch.save(
        {
            "state": network.state_dict(),
            "tags": list(tags),
            "features": int(network.net[0].in_channels),
            "thresholds": thresholds,
        },
        path,
    )


def load_sequence_model(path):
    import torch

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    network = _Network(int(blob["features"]), len(blob["tags"]))
    network.load_state_dict(blob["state"])
    network.eval()
    return SequenceClassifier(network, blob["tags"]), list(blob["tags"]), blob.get("thresholds") or {}


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _Network:
    """시간을 따라 보는 합성곱 두 층. torch 가 없을 때도 이 파일은 열립니다."""

    def __new__(cls, features, class_count):
        import torch
        from torch import nn

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(features, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.fc = nn.Linear(64, class_count)

            def forward(self, sequence):
                # (묶음, 시간, 좌표) 를 (묶음, 좌표, 시간) 으로 두고 시간을 따라 봅니다.
                hidden = self.net(sequence.transpose(1, 2)).squeeze(-1)
                return self.fc(hidden)

        return Network()
