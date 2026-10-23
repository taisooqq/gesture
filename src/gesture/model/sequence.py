"""딥러닝 동작 모델. 프레임을 앞에서부터 순서대로 읽고, 마지막 장까지 읽은 상태로 판단합니다.

합성곱은 지금 장과 바로 앞 두 장만 봅니다. 뒤를 0으로 채운 칸은 앞쪽 판단에 섞이지 않고,
마지막으로 손이 있던 장의 결과만 점수 층에 들어갑니다. 평균은 내지 않습니다.
"""

import numpy as np

from gesture.model.motion import FEATURE_KIND, Augmenter, chain, features

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
            # 앞·끝 멈춘 손을 무동작으로 자주 넣으므로 가중치도 더 준다.
            counts[idle] += 2.0
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1.0) / len(tags), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    augmenter = Augmenter(settings, rng)
    batch = 64
    for _epoch in range(int(settings.seq_epochs)):
        xs, ys = _epoch_samples(clips, augmenter, idle, rng, tags, settings)
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


def _epoch_samples(clips, augmenter, idle, rng, tags, settings):
    """한 회차의 학습 묶음. 매번 다르게 바꾼 좌표와, 멈춘 손 장면을 섞습니다."""
    hold_pose = {index for index, name in enumerate(tags) if name in settings.hold_pose_tags}
    xs = []
    ys = []
    for label, joints, depth, aspect, moving in clips:
        xs.append(augmenter.moved(joints, depth, aspect))
        ys.append(label)
        if moving and idle is not None:
            # 맨 앞·맨 끝 자세로 멈춘 손을 둘 다 무동작으로 넣는다. 인식에서 무동작을 더 세게 본다.
            for index in (0, joints.shape[0] - 1):
                xs.append(augmenter.held(joints, depth, aspect, index))
                ys.append(idle)
        elif not moving and rng.random() < 0.25:
            held = augmenter.held(joints, depth, aspect, int(rng.integers(0, joints.shape[0])))
            xs.append(held)
            # 스크롤만 멈춘 손을 그 동작으로 둔다. 우클릭 등은 멈춘 손을 무동작으로 둔다.
            if label in hold_pose:
                ys.append(label)
            elif idle is not None:
                ys.append(idle)
    _add_chained_clicks(clips, augmenter, rng, tags, settings, xs, ys)
    return xs, ys


# 연속 클릭(더블클릭)용 학습 묶음.
# "떼기 끝 몇 장 + 누르기" 는 누르기, "누르기 끝 몇 장 + 떼기" 는 떼기로 넣는다.
# 모델이 아는 동작은 그대로 누르기·떼기 둘이다. 빠르게 이어지는 두 번째 누르기·떼기를 잘 알아보게 한다.
# 화면 비율이 다른 영상끼리는 손 모양이 찌그러지므로 같은 비율끼리만 붙인다.
def _add_chained_clicks(clips, augmenter, rng, tags, settings, xs, ys):
    ratio = float(getattr(settings, "chain_click_ratio", 0.0))
    if ratio <= 0 or "left_click_on" not in tags or "left_click_off" not in tags:
        return
    on = tags.index("left_click_on")
    off = tags.index("left_click_off")
    pools = {on: {}, off: {}}
    for clip in clips:
        if clip[0] in pools:
            pools[clip[0]].setdefault(round(float(clip[3]), 2), []).append(clip)
    low, high = (int(value) for value in settings.chain_tail_frames)
    window = int(settings.seq_window)
    for label, before in ((on, off), (off, on)):
        for _label, joints, depth, aspect, _moving in [clip for group in pools[label].values() for clip in group]:
            if rng.random() >= ratio:
                continue
            partners = pools[before].get(round(float(aspect), 2))
            if not partners:
                continue
            prefix = partners[int(rng.integers(len(partners)))]
            tail = int(rng.integers(low, high + 1))
            joined_j, joined_d = chain(prefix[1], prefix[2], joints, depth, tail, window)
            xs.append(augmenter.moved(joined_j, joined_d, aspect, edges=False))
            ys.append(label)


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
