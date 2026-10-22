"""학습 영상마다 손 좌표를 한 번만 뽑아 저장합니다. 여러 영상은 코어마다 나눠 뽑습니다."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from gesture.recorder import open_video

CACHE_VERSION = 1
_WRIST = 0
_MIDDLE_MCP = 9
_HANDS = {}


def video_points(settings, videos):
    """영상마다 (프레임별 손 좌표 목록, 화면 가로/세로). 손이 없는 프레임은 None 입니다."""
    complexity = int(settings.hand_complexity)
    found = {}
    missing = []
    for video in videos:
        cached = _read_cache(settings, video, complexity)
        if cached is None:
            missing.append(video)
        else:
            found[video] = cached
    for video, arrays in _extract_many(settings, missing, complexity):
        _write_cache(settings, video, complexity, arrays)
        found[video] = arrays
    return {video: (to_points(*found[video][:3]), float(found[video][3])) for video in videos}


def to_points(joints, depth, present):
    """저장한 배열을 HandAnalyzer 와 같은 손 좌표 목록으로 바꿉니다."""
    points = []
    for index in range(len(present)):
        if not present[index]:
            points.append(None)
            continue
        frame_joints = joints[index].astype(np.float32)
        wrist = frame_joints[_WRIST]
        scale = float(np.linalg.norm(frame_joints[_MIDDLE_MCP] - wrist))
        if scale < 1e-4:
            points.append(None)
            continue
        points.append(
            {"joints": frame_joints, "wrist": wrist, "scale": scale, "depth": depth[index].astype(np.float32)}
        )
    return points


def _extract_many(settings, videos, complexity):
    if not videos:
        return []
    workers = settings.landmark_workers or max(1, (os.cpu_count() or 2) - 1)
    workers = min(int(workers), len(videos))
    jobs = [(str(video), complexity) for video in videos]
    if workers > 1 and len(videos) >= 4:
        try:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                results = list(pool.map(_extract_job, jobs, chunksize=4))
            return list(zip(videos, results))
        except Exception:
            pass
    return [(video, _extract_job(job)) for video, job in zip(videos, jobs)]


def _extract_job(job):
    """한 영상의 모든 프레임에서 손을 따로 찾습니다. 학습과 같은 static 모드입니다."""
    path, complexity = job
    hands = _hands(complexity)
    cap = open_video(path)
    joints = []
    depth = []
    present = []
    aspect = 16 / 9
    first = True
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if first:
            height, width = frame.shape[:2]
            aspect = width / max(height, 1)
            first = False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks[0].landmark) == 21:
            landmark = results.multi_hand_landmarks[0].landmark
            joints.append([[point.x, point.y] for point in landmark])
            depth.append([point.z for point in landmark])
            present.append(True)
        else:
            joints.append(np.zeros((21, 2)))
            depth.append(np.zeros(21))
            present.append(False)
    cap.release()
    if not present:
        return np.zeros((0, 21, 2), np.float32), np.zeros((0, 21), np.float32), np.zeros(0, bool), aspect
    return (
        np.asarray(joints, dtype=np.float32),
        np.asarray(depth, dtype=np.float32),
        np.asarray(present, dtype=bool),
        float(aspect),
    )


def _hands(complexity):
    if complexity not in _HANDS:
        import mediapipe as mp

        _HANDS[complexity] = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _HANDS[complexity]


def _cache_file(settings, video):
    relative = video.resolve().relative_to(settings.video_dir.resolve())
    return settings.landmark_dir / relative.with_suffix(".npz")


def _stamp(video, complexity):
    import mediapipe as mp

    stat = video.stat()
    return f"{CACHE_VERSION}:{stat.st_size}:{stat.st_mtime_ns}:{complexity}:{mp.__version__}"


def _read_cache(settings, video, complexity):
    try:
        target = _cache_file(settings, video)
    except ValueError:
        return None
    if not target.exists():
        return None
    try:
        with np.load(target, allow_pickle=False) as blob:
            if str(blob["stamp"]) != _stamp(video, complexity):
                return None
            return blob["joints"], blob["depth"], blob["present"], float(blob["aspect"])
    except (OSError, KeyError, ValueError):
        return None


def _write_cache(settings, video, complexity, arrays):
    try:
        target = _cache_file(settings, video)
    except ValueError:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    joints, depth, present, aspect = arrays
    temp = target.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp,
        stamp=np.array(_stamp(video, complexity)),
        joints=joints,
        depth=depth,
        present=present,
        aspect=np.array(aspect, dtype=np.float32),
    )
    os.replace(temp, target)
