"""중앙 영역을 자르고, 미디어파이프 관절을 학습 숫자로 바꿉니다."""

from collections import OrderedDict

import cv2
import mediapipe as mp
import numpy as np

from gesture.config import settings as default_settings


class HandAnalyzer:
    """중앙 영역 자르기와 관절 숫자."""

    def __init__(self, settings=None):
        self.settings = settings or default_settings
        self._hands = None
        self.hands_error = None
        self._point_cache = OrderedDict()

    # 학습 영상 2. 가로·세로 약 55%의 정중앙만 남긴다.
    # 손 위치를 따라가지 않는다. 얼굴이 파일과 인식에 들어가지 않게 하려는 고정 영역이다.
    def crop(self, frame):
        h, w = frame.shape[:2]
        # 화면의 약 55%만 남긴다. 얼굴이 저장·인식에 들어가지 않게 하려는 영역이다.
        rw = int(w * self.settings.roi_scale)
        rh = int(h * self.settings.roi_scale)
        x1 = (w - rw) // 2
        y1 = (h - rh) // 2
        return frame[y1 : y1 + rh, x1 : x1 + rw].copy(), (x1, y1, rw, rh)

    def skin_mask(self, bgr):
        """예전 모델용. 피부색만 남긴 흑백 손 도장을 만듭니다."""
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def extract(self, bgr):
        """예전 모델용. 한 장의 윤곽에서 숫자 8개를 만듭니다. 손이 없으면 None입니다."""
        mask = self.skin_mask(bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape[:2]
        frame_area = float(h * w)
        if not contours:
            return None, mask
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < frame_area * self.settings.min_hand_ratio:
            return None, mask
        _x, _y, bw, bh = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) + 1e-6
        peri = float(cv2.arcLength(contour, True)) + 1e-6
        feat = np.array(
            [
                area / frame_area,
                bw / (bh + 1e-6),
                area / hull_area,
                4.0 * np.pi * area / (peri * peri),
                float(self._defect_count(contour)),
                *self._hu3(contour),
            ],
            dtype=np.float32,
        )
        return feat, mask

    def _defect_count(self, contour):
        """윤곽에서 손가락 사이처럼 깊게 들어간 홈의 개수."""
        if len(contour) < 4:
            return 0
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            return 0
        try:
            defects = cv2.convexityDefects(contour, hull_idx)
        except cv2.error:
            return 0
        if defects is None:
            return 0
        defects = np.asarray(defects)
        if defects.ndim == 3:
            defects = defects[:, 0, :]
        count = 0
        for row in defects:
            if float(row[3]) / 256.0 > 18:
                count += 1
        return count

    def _hu3(self, contour):
        """윤곽 모양을 요약하는 형태값 3개."""
        hu = cv2.HuMoments(cv2.moments(contour)).flatten()
        return [
            float(-np.sign(value) * np.log10(abs(value) + 1e-12))
            for value in hu[:3]
        ]

    # 학습 숫자는 여기다. 인식은 camera.py _predict 가 features_for 로 같은 숫자를 만든다.
    # 손목을 원점으로 두고, 다섯 손가락 끝만 쓴다. 엄지, 검지, 중지, 약지, 새끼 순이다.
    # 끝마다 x, y, dx, dy. x, y는 손목을 뺀 위치, dx, dy는 1초 동안 그 끝이 손목 기준으로 움직인 양이다.
    # 손 크기는 손목에서 중지 뿌리까지로 나눈다. z는 넣지 않는다.
    FEATURE_KIND = "fingertips"
    WINDOW_KINDS = ("fingertips", "mediapipe_joints", "mediapipe")
    _TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
    _TIPS = (4, 8, 12, 16, 20)
    MOTION_FEATURES = tuple(
        f"{name}_{axis}"
        for name in _TIP_NAMES
        for axis in ("x", "y", "dx", "dy")
    )
    LANDMARK_COUNT = 21
    _WRIST = 0
    _MIDDLE_MCP = 9

    def close(self):
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def window_features(self, frames):
        """1초 묶음을 다섯 손가락 끝의 위치와 이동으로 바꿉니다. 손이 거의 없으면 None입니다."""
        valid, scale = self._valid_frames(frames)
        if valid is None:
            return None
        rows = []
        for tip in self._TIPS:
            # 손목을 뺀 끝 위치. 화면 어디에 손이 있는지는 빠진다.
            placed = np.stack([(item["joints"][tip] - item["wrist"]) / scale for item in valid])
            # 1초 중간의 위치와, 처음에서 끝까지의 이동.
            rows.append(np.concatenate([np.median(placed, axis=0), placed[-1] - placed[0]]))
        return np.concatenate(rows).astype(np.float32)

    def _joint_features(self, frames):
        """예전 모델용. 관절 21개의 위치와 화면 이동을 한 줄로 만듭니다."""
        valid, scale = self._valid_frames(frames)
        if valid is None:
            return None
        placed = [(item["joints"] - item["wrist"]) / scale for item in valid]
        shape = np.median(np.stack(placed), axis=0)
        motion = (valid[-1]["joints"] - valid[0]["joints"]) / scale
        return np.concatenate([shape, motion], axis=1).reshape(-1).astype(np.float32)

    def _valid_frames(self, frames):
        """손이 잡힌 장면과 손 크기를 고릅니다. 너무 적으면 (None, None)입니다."""
        points = [self._finger_points(frame) for frame in frames]
        valid = [item for item in points if item is not None]
        if len(valid) < max(3, len(frames) // 2):
            return None, None
        scale = float(np.median([item["scale"] for item in valid])) + 1e-6
        return valid, scale

    def features_for(self, frames, kind):
        """고른 모델이 학습할 때 본 것과 같은 숫자 한 줄을 만듭니다."""
        # 다섯 손가락 끝. 지금 학습 버튼이 만드는 모델이다.
        if kind == self.FEATURE_KIND:
            return self.window_features(frames)
        # 관절 21개 전부.
        if kind == "mediapipe_joints":
            return self._joint_features(frames)
        # 엄지·검지·중지 거리와 검지 이동 5개.
        if kind == "mediapipe":
            return self._summary_features(frames)
        # 좌표 종류가 없는 예전 모델은 한 장의 윤곽 숫자 8개다.
        if not frames:
            return None
        feat, _mask = self.extract(frames[-1])
        return feat

    def _summary_features(self, frames):
        """1초 묶음을 손가락 거리 2개와 검지 이동 3개로 바꿉니다."""
        points = [self._finger_points(frame) for frame in frames]
        valid = [item for item in points if item is not None]
        if len(valid) < max(3, len(frames) // 2):
            return None
        scale = float(np.median([item["scale"] for item in valid])) + 1e-6
        thumb = 4
        index = 8
        middle = 12
        thumb_index = min(
            np.linalg.norm(item["joints"][thumb] - item["joints"][index]) for item in valid
        ) / scale
        thumb_middle = min(
            np.linalg.norm(item["joints"][thumb] - item["joints"][middle]) for item in valid
        ) / scale
        index_path = [(item["joints"][index] - item["wrist"]) / scale for item in valid]
        delta = index_path[-1] - index_path[0]
        return np.array(
            [
                thumb_index,
                thumb_middle,
                float(delta[0]),
                float(delta[1]),
                self._turn(index_path),
            ],
            dtype=np.float32,
        )

    def _turn(self, path):
        """검지 경로가 중심 주변을 얼마나 돌았는지."""
        if len(path) < 3:
            return 0.0
        kept = [path[0]]
        for point in path[1:]:
            if np.linalg.norm(point - kept[-1]) >= 0.08:
                kept.append(point)
        if len(kept) < 3:
            return 0.0
        center = np.mean(np.vstack(kept), axis=0)
        vectors = [point - center for point in kept]
        total = 0.0
        for prev, curr in zip(vectors, vectors[1:]):
            cross = float(prev[0] * curr[1] - prev[1] * curr[0])
            dot = float(prev[0] * curr[0] + prev[1] * curr[1])
            total += float(np.arctan2(cross, dot))
        return abs(total)

    def _finger_points(self, bgr):
        """한 장의 중앙 화면에서 관절 21개를 찾습니다. 손이 없으면 None입니다."""
        key = self._frame_key(bgr)
        if key in self._point_cache:
            self._point_cache.move_to_end(key)
            return self._point_cache[key]
        points = self._read_landmarks(bgr)
        self._point_cache[key] = points
        if len(self._point_cache) > 256:
            self._point_cache.popitem(last=False)
        return points

    def _frame_key(self, bgr):
        """같은 화면을 묶음이 겹칠 때 다시 돌리지 않으려고, 배열과 내용으로 구분합니다."""
        sample = np.ascontiguousarray(bgr[::16, ::16])
        return (id(bgr), sample.shape, hash(sample.tobytes()))

    def _read_landmarks(self, bgr):
        """프레임마다 따로 관절을 찾습니다. 화면 비율 좌표이고, 픽셀 좌표는 쓰지 않습니다."""
        hands = self._ensure_hands()
        if hands is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        landmark = results.multi_hand_landmarks[0].landmark
        # 화면 가로·세로에 대한 0~1 좌표. 픽셀 좌표와 z는 여기서 버린다.
        joints = np.array([[point.x, point.y] for point in landmark], dtype=np.float32)
        if len(joints) != self.LANDMARK_COUNT:
            return None
        wrist = joints[self._WRIST]
        scale = float(np.linalg.norm(joints[self._MIDDLE_MCP] - wrist))
        if scale < 1e-4:
            return None
        return {"joints": joints, "wrist": wrist, "scale": scale}

    def _ensure_hands(self):
        """추적 없이 한 장씩 찾습니다. 영상 파일이 바뀌어도 이전 손이 남지 않습니다."""
        if self._hands is not None:
            return self._hands
        if self.hands_error:
            return None
        try:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5,
            )
        except Exception as exc:
            self.hands_error = f"손 좌표 모델을 열지 못했습니다. {exc}"
            return None
        return self._hands
