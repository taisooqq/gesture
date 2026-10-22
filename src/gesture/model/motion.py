"""딥러닝이 보는 숫자. 학습과 인식이 이 파일의 같은 함수로 만듭니다.

장마다 자세(손목 기준 관절 21개, 깊이, 굽은 각)와 이전 장에서 변한 양,
손목이 화면에서 움직인 양과 손 크기 변화가 들어갑니다. 첫 장의 변화는 0입니다.
"""

import numpy as np

FEATURE_KIND = "pose_motion_v1"
_WRIST = 0
_MIDDLE_MCP = 9
_BENDS = ((3, 2, 4), (6, 5, 8), (10, 9, 12), (14, 13, 16), (18, 17, 20))


def stack_points(points):
    """손이 잡힌 장만 (장, 21, 2), (장, 21) 배열로 쌓습니다."""
    valid = [item for item in points if item is not None]
    if not valid:
        return None, None
    joints = np.stack([item["joints"] for item in valid]).astype(np.float32)
    depth = np.stack([item["depth"] for item in valid]).astype(np.float32)
    return joints, depth


def features(joints, depth, aspect):
    """화면 비율 좌표를 장마다의 숫자 (장, 139) 로 바꿉니다."""
    joints, depth = _square(joints, depth, aspect)
    return _features(joints, depth)


def _square(joints, depth, aspect):
    """가로 1과 세로 1의 길이를 같게 맞춥니다. 16:9 화면에서 각과 회전이 찌그러지지 않습니다."""
    joints = joints.astype(np.float32).copy()
    joints[..., 0] *= float(aspect)
    return joints, depth.astype(np.float32) * float(aspect)


def _features(joints, depth):
    count = joints.shape[0]
    wrist = joints[:, _WRIST]
    scale = np.linalg.norm(joints[:, _MIDDLE_MCP] - wrist, axis=-1)
    scale = np.maximum(scale, 1e-4)
    xy = (joints - wrist[:, None, :]) / scale[:, None, None]
    z = (depth - depth[:, _WRIST : _WRIST + 1]) / scale[:, None]
    angles = []
    for joint, left, right in _BENDS:
        before = joints[:, left] - joints[:, joint]
        after = joints[:, right] - joints[:, joint]
        lengths = np.linalg.norm(before, axis=-1) * np.linalg.norm(after, axis=-1)
        cosine = np.sum(before * after, axis=-1) / np.maximum(lengths, 1e-12)
        degrees = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        angles.append(np.where(lengths < 1e-12, 180.0, degrees) / 180.0)
    pose = np.concatenate([xy.reshape(count, -1), z, np.stack(angles, axis=1)], axis=1)
    reference = float(np.median(scale))
    change = np.zeros_like(pose)
    change[1:] = pose[1:] - pose[:-1]
    move = np.zeros((count, 2), dtype=np.float32)
    move[1:] = (wrist[1:] - wrist[:-1]) / reference
    grow = np.zeros((count, 1), dtype=np.float32)
    grow[1:, 0] = np.log(scale[1:] / scale[:-1])
    return np.concatenate([pose, change, move, grow], axis=1).astype(np.float32)


def windows(points, length, stride=5):
    """인식 1초와 같은 길이로 자릅니다. 마지막 묶음은 영상 끝에 맞춰 꼬리를 버리지 않습니다."""
    if len(points) <= length:
        return [points] if points else []
    starts = list(range(0, len(points) - length + 1, stride))
    if starts[-1] != len(points) - length:
        starts.append(len(points) - length)
    return [points[start : start + length] for start in starts]


class Augmenter:
    """학습 때만 씁니다. 매 학습 회차마다 새로 바꾼 좌표를 만듭니다."""

    def __init__(self, settings, rng):
        self.settings = settings
        self.rng = rng
        self.window = int(settings.seq_window)
        self.min_frames = int(settings.seq_min_frames)

    def moved(self, joints, depth, aspect):
        """사다리꼴, 회전, 속도, 앞뒤로 멈춘 장면, 흔들림을 임의로 섞습니다."""
        rng = self.rng
        if rng.random() < 0.2:
            return features(joints, depth, aspect)
        joints = self._trapezoid(joints, _pick(rng, self.settings.augment_trapezoids))
        joints, depth = _square(joints, depth, aspect)
        joints = _rotate(joints, _pick(rng, self.settings.augment_rotations))
        if rng.random() < 0.5:
            joints, depth = _warp(joints, depth, _pick(rng, self.settings.augment_speeds))
        joints, depth = self._still_edges(joints, depth)
        joints, depth = self._jitter(joints, depth)
        return _features(joints, depth)

    def held(self, joints, depth, aspect, index):
        """한 자세로 멈춘 손. 움직여야 하는 동작이면 무동작, 자세 동작이면 그 동작입니다."""
        rng = self.rng
        count = int(rng.integers(self.min_frames, self.window + 1))
        joints = np.repeat(joints[index : index + 1], count, axis=0)
        depth = np.repeat(depth[index : index + 1], count, axis=0)
        joints = self._trapezoid(joints, _pick(rng, self.settings.augment_trapezoids))
        joints, depth = _square(joints, depth, aspect)
        joints = _rotate(joints, _pick(rng, self.settings.augment_rotations))
        joints, depth = self._jitter(joints, depth)
        return _features(joints, depth)

    def _trapezoid(self, joints, amount):
        if amount == 0.0:
            return joints
        shaped = joints.copy()
        scale_x = (1.0 + amount) - (2.0 * amount) * shaped[..., 1]
        shaped[..., 0] = 0.5 + (shaped[..., 0] - 0.5) * scale_x
        return shaped

    def _still_edges(self, joints, depth):
        """동작 앞뒤에 손을 멈춘 장을 붙입니다. 인식 1초에는 동작 앞뒤 자세가 같이 들어옵니다."""
        count = joints.shape[0]
        extra = self.window - count
        if extra <= 0 or self.rng.random() < 0.3:
            return joints, depth
        total = int(self.rng.integers(0, extra + 1))
        before = int(self.rng.integers(0, total + 1))
        after = total - before
        index = [0] * before + list(range(count)) + [count - 1] * after
        return joints[index], depth[index]

    def _jitter(self, joints, depth):
        scale = float(np.median(np.linalg.norm(joints[:, _MIDDLE_MCP] - joints[:, _WRIST], axis=-1)))
        sigma = float(self.settings.augment_jitter) * scale
        if sigma <= 0:
            return joints, depth
        joints = joints + self.rng.normal(0.0, sigma, size=joints.shape).astype(np.float32)
        depth = depth + self.rng.normal(0.0, sigma, size=depth.shape).astype(np.float32)
        return joints, depth


def _pick(rng, values):
    """설정한 값들의 최솟값과 최댓값 사이에서 고릅니다."""
    values = [float(value) for value in values]
    if not values:
        return 0.0
    low, high = min(values), max(values)
    return float(rng.uniform(low, high))


def _rotate(joints, degrees):
    if degrees == 0.0:
        return joints
    theta = np.deg2rad(degrees)
    cosine, sine = float(np.cos(theta)), float(np.sin(theta))
    rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float32)
    wrist = joints[:, _WRIST : _WRIST + 1]
    return ((joints - wrist) @ rotation + wrist).astype(np.float32)


def _warp(joints, depth, speed):
    """더 빠르거나 느린 같은 동작. 빠르면 일찍 끝나고 끝 자세로 남습니다."""
    count = joints.shape[0]
    if count < 2 or speed <= 0:
        return joints, depth
    positions = np.clip(np.arange(count, dtype=np.float64) * float(speed), 0, count - 1)
    left = np.floor(positions).astype(int)
    right = np.minimum(left + 1, count - 1)
    weight = (positions - left).astype(np.float32)
    joints = (1.0 - weight)[:, None, None] * joints[left] + weight[:, None, None] * joints[right]
    depth = (1.0 - weight)[:, None] * depth[left] + weight[:, None] * depth[right]
    return joints.astype(np.float32), depth.astype(np.float32)
