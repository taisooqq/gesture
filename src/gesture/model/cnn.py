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
        length = getattr(self.network, "length", None)
        lengths = None
        if length:
            if tensor.shape[1] > length:
                tensor = tensor[:, -int(length) :]
            lengths = torch.tensor([int(tensor.shape[1])], dtype=torch.int64)
        with torch.no_grad():
            logits = self.network(tensor, lengths)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba[0]


def train_sequence_model(sequences, labels, class_count):
    """좌표 시계열로 작은 1D CNN을 맞춥니다. 짧은 영상은 가장 긴 묶음 길이까지 뒤를 0으로 채웁니다."""
    import torch
    from torch import nn

    items = _as_sequences(sequences)
    features = int(items[0].shape[-1])
    width = max(int(item.shape[0]) for item in items)
    network = _Network(features, class_count, width)
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
            padded, lengths = _pad([items[index] for index in chosen], width)
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


def _pad(items, width=None):
    """짧은 시계열 뒤를 0으로 채워 한 묶음으로 만듭니다."""
    import torch

    if width is None:
        width = max(int(item.shape[0]) for item in items)
    width = int(width)
    features = int(items[0].shape[-1])
    padded = torch.zeros((len(items), width, features), dtype=torch.float32)
    lengths = torch.zeros(len(items), dtype=torch.int64)
    for index, item in enumerate(items):
        tensor = torch.from_numpy(np.asarray(item, dtype=np.float32))
        count = min(int(tensor.shape[0]), width)
        if count:
            padded[index, :count] = tensor[-count:]
        lengths[index] = count
    return padded, lengths


def save_sequence_model(path, network, tags, thresholds):
    import torch

    torch.save(
        {
            "state": network.state_dict(),
            "tags": list(tags),
            "features": int(network.net[0].in_channels),
            "length": int(network.length) if getattr(network, "length", None) else None,
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
    network = _Network(int(blob["features"]), len(blob["tags"]), blob.get("length"))
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

    def __new__(cls, features, class_count, length=None):
        import torch
        from torch import nn

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.length = int(length) if length else None
                self.net = nn.Sequential(
                    nn.Conv1d(features, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                hidden = 64 if self.length is None else 64 * self.length
                self.fc = nn.Linear(hidden, class_count)

            def forward(self, sequence, lengths=None):
                # (묶음, 시간, 좌표) 를 (묶음, 좌표, 시간) 으로 두고 시간을 따라 봅니다.
                # 길이가 없으면 이미 저장된 모델이라 시간 평균을 그대로 씁니다.
                if self.length is None:
                    hidden = self.net(sequence.transpose(1, 2))
                    return self.fc(_masked_mean(hidden, lengths))
                target = self.length
                steps = sequence.shape[1]
                if steps > target:
                    sequence = sequence[:, -target:]
                    steps = target
                    if lengths is not None:
                        lengths = lengths.clamp(max=target)
                if steps < target:
                    pad = sequence.new_zeros(sequence.shape[0], target - steps, sequence.shape[2])
                    sequence = torch.cat([sequence, pad], dim=1)
                hidden = self.net(sequence.transpose(1, 2))
                if lengths is not None:
                    index = torch.arange(target, device=hidden.device)
                    used = lengths.clamp(min=0, max=target)
                    mask = (index.unsqueeze(0) < used.unsqueeze(1)).to(hidden.dtype).unsqueeze(1)
                    hidden = hidden * mask
                ordered = hidden.transpose(1, 2).reshape(hidden.shape[0], target * hidden.shape[1])
                return self.fc(ordered)

        return Network()


def _masked_mean(hidden, lengths):
    """이미 저장된 모델용. 빈 칸을 빼고 시간 축을 평균냅니다."""
    import torch

    if lengths is None:
        return hidden.mean(dim=-1)
    steps = hidden.shape[-1]
    index = torch.arange(steps, device=hidden.device)
    mask = (index.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1)
    denom = lengths.clamp(min=1).to(hidden.dtype).unsqueeze(1)
    return (hidden * mask.to(hidden.dtype)).sum(dim=-1) / denom
