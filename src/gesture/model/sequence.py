"""딥러닝 동작 모델. 프레임을 앞에서부터 순서대로 읽고, 마지막 장까지 읽은 상태로 판단합니다.

합성곱은 지금 장과 바로 앞 두 장만 봅니다. 뒤를 0으로 채운 칸은 앞쪽 판단에 섞이지 않고,
마지막으로 손이 있던 장의 결과만 점수 층에 들어갑니다. 평균은 내지 않습니다.
"""

import numpy as np

from gesture.model.motion import FEATURE_KIND, Augmenter, features

VERSION = 2


class MotionModel:
    """랜덤포레스트처럼 predict_proba 와 classes_ 를 제공합니다."""

    def __init__(self, network, tags, window):
        self.network = network
        self.tags = list(tags)
        self.window = int(window)
        self.classes_ = np.arange(len(tags))

    def predict_proba(self, sequence):
        import torch

        sequence = np.asarray(sequence, dtype=np.float32)[-self.window :]
        tensor = torch.from_numpy(sequence).unsqueeze(0)
        lengths = torch.tensor([sequence.shape[0]], dtype=torch.int64)
        self.network.eval()
        with torch.no_grad():
            logits = self.network(tensor, lengths)
            return torch.softmax(logits, dim=1)[0].cpu().numpy()


def live_features(points, aspect):
    """인식용. 손이 잡힌 장의 숫자입니다. 증강은 하지 않습니다."""
    from gesture.model.motion import stack_points

    joints, depth = stack_points(points)
    if joints is None:
        return None
    return features(joints, depth, aspect)


def train_motion_model(clips, tags, settings, seed=0):
    """clips 는 (태그 번호, 관절, 깊이, 화면 비율, 움직이는 동작인지) 목록입니다."""
    import torch
    from torch import nn

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idle = tags.index("idle") if "idle" in tags else None
    plain = np.concatenate([features(joints, depth, aspect) for _label, joints, depth, aspect, _moving in clips])
    network = _network(plain.shape[1], len(tags), plain.mean(axis=0), plain.std(axis=0))
    device = _device()
    network.to(device)
    counts = np.zeros(len(tags), dtype=np.float64)
    for label, _joints, _depth, _aspect, moving in clips:
        counts[label] += 1.0
        if moving and idle is not None:
            counts[idle] += 0.5
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1.0) / len(tags), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    augmenter = Augmenter(settings, rng)
    batch = 64
    for _epoch in range(int(settings.seq_epochs)):
        xs, ys = _epoch_samples(clips, augmenter, idle, rng)
        order = rng.permutation(len(xs))
        network.train()
        for start in range(0, len(order), batch):
            chosen = order[start : start + batch]
            padded, lengths = _pad([xs[index] for index in chosen])
            target = torch.from_numpy(np.asarray([ys[index] for index in chosen], dtype=np.int64))
            optimizer.zero_grad()
            logits = network(padded.to(device), lengths.to(device))
            loss = loss_fn(logits, target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
    network.to("cpu")
    network.eval()
    return MotionModel(network, tags, settings.seq_window)


def _epoch_samples(clips, augmenter, idle, rng):
    """한 회차의 학습 묶음. 매번 다르게 바꾼 좌표와, 멈춘 손 장면을 섞습니다."""
    xs = []
    ys = []
    for label, joints, depth, aspect, moving in clips:
        xs.append(augmenter.moved(joints, depth, aspect))
        ys.append(label)
        if moving and idle is not None and rng.random() < 0.5:
            index = 0 if rng.random() < 0.5 else joints.shape[0] - 1
            xs.append(augmenter.held(joints, depth, aspect, index))
            ys.append(idle)
        elif not moving and rng.random() < 0.15:
            xs.append(augmenter.held(joints, depth, aspect, int(rng.integers(0, joints.shape[0]))))
            ys.append(label)
    return xs, ys


def _pad(items):
    import torch

    width = max(item.shape[0] for item in items)
    size = items[0].shape[1]
    padded = np.zeros((len(items), width, size), dtype=np.float32)
    lengths = np.zeros(len(items), dtype=np.int64)
    for index, item in enumerate(items):
        padded[index, : item.shape[0]] = item
        lengths[index] = item.shape[0]
    return torch.from_numpy(padded), torch.from_numpy(lengths)


def save_motion_model(path, model, thresholds):
    import torch

    network = model.network
    torch.save(
        {
            "version": VERSION,
            "kind": FEATURE_KIND,
            "state": network.state_dict(),
            "tags": model.tags,
            "features": int(network.conv1.in_channels),
            "hidden": int(network.conv1.out_channels),
            "window": model.window,
            "thresholds": thresholds,
        },
        path,
    )


def load_motion_model(path):
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    features_count = int(blob["features"])
    network = _network(
        features_count,
        len(blob["tags"]),
        np.zeros(features_count, np.float32),
        np.ones(features_count, np.float32),
        int(blob.get("hidden", 96)),
    )
    network.load_state_dict(blob["state"])
    network.eval()
    model = MotionModel(network, blob["tags"], int(blob.get("window", 15)))
    return model, list(blob["tags"]), blob.get("thresholds") or {}


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _network(features_count, class_count, mean, std, hidden=96):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.as_tensor(np.asarray(mean, np.float32)))
            self.register_buffer("std", torch.as_tensor(np.maximum(np.asarray(std, np.float32), 1e-3)))
            self.conv1 = nn.Conv1d(features_count, hidden, kernel_size=3)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3)
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden, class_count)

        def forward(self, sequence, lengths):
            # 합성곱은 앞쪽만 채워 지금 장과 그 앞 두 장만 본다.
            hidden_seq = ((sequence - self.mean) / self.std).transpose(1, 2)
            hidden_seq = F.relu(self.conv1(F.pad(hidden_seq, (2, 0))))
            hidden_seq = F.relu(self.conv2(F.pad(hidden_seq, (2, 0))))
            output, _state = self.gru(hidden_seq.transpose(1, 2))
            last = (lengths.to(output.device) - 1).clamp(min=0)
            picked = output[torch.arange(output.shape[0], device=output.device), last]
            return self.fc(self.drop(picked))

    return Network()
