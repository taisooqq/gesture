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
    """좌표 시계열로 작은 1D CNN을 맞춥니다. 길이가 다른 영상은 빈 칸으로 채웁니다."""
    import torch
    from torch import nn

    items = _as_sequences(sequences)
    features = int(items[0].shape[-1])
    network = _Network(features, class_count)
    device = _device()
    network.to(device)
    ys = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(ys, minlength=class_count).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    batch = 128
    network.train()
    for _epoch in range(20):
        order = torch.randperm(len(items))
        for start in range(0, len(items), batch):
            chosen = order[start : start + batch].tolist()
            padded, lengths = _pad([items[index] for index in chosen])
            target = torch.from_numpy(ys[chosen])
            optimizer.zero_grad()
            logits = network(padded.to(device), lengths.to(device))
            loss = loss_fn(logits, target.to(device))
            loss.backward()
            optimizer.step()
    network.to("cpu")
    network.eval()
    return network


def _as_sequences(sequences):
    """길이별로 쌓인 배열이거나, 이미 같은 길이로 묶인 배열을 목록으로 바꿉니다."""
    if isinstance(sequences, np.ndarray) and sequences.ndim == 3:
        return [sequences[index] for index in range(sequences.shape[0])]
    return [np.asarray(item, dtype=np.float32) for item in sequences]


def _pad(items):
    """짧은 시계열 뒤를 0으로 채워 한 묶음으로 만듭니다."""
    import torch

    width = max(int(item.shape[0]) for item in items)
    features = int(items[0].shape[-1])
    padded = torch.zeros((len(items), width, features), dtype=torch.float32)
    lengths = torch.zeros(len(items), dtype=torch.int64)
    for index, item in enumerate(items):
        tensor = torch.from_numpy(np.asarray(item, dtype=np.float32))
        padded[index, : tensor.shape[0]] = tensor
        lengths[index] = tensor.shape[0]
    return padded, lengths


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
                )
                self.fc = nn.Linear(64, class_count)

            def forward(self, sequence, lengths=None):
                # (묶음, 시간, 좌표) 를 (묶음, 좌표, 시간) 으로 두고 시간을 따라 봅니다.
                # 짧은 영상 뒤의 빈 칸은 평균에서 뺍니다.
                hidden = self.net(sequence.transpose(1, 2))
                if lengths is None:
                    pooled = hidden.mean(dim=-1)
                else:
                    steps = hidden.shape[-1]
                    index = torch.arange(steps, device=hidden.device)
                    mask = (index.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1)
                    denom = lengths.clamp(min=1).to(hidden.dtype).unsqueeze(1)
                    pooled = (hidden * mask.to(hidden.dtype)).sum(dim=-1) / denom
                return self.fc(pooled)

        return Network()
