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
        self._hands_static = None
        self._static_image = False
        self.hands_error = None
        self._point_cache = OrderedDict()

    # 등록 녹화만 가로·세로 약 55%의 정중앙을 남긴다. 인식은 화면 전체를 본다.
    def crop(self, frame):
        x1, y1, rw, rh = self.center_box(frame, self.settings.roi_scale)
        return frame[y1 : y1 + rh, x1 : x1 + rw].copy(), (x1, y1, rw, rh)

    def center_box(self, frame, scale):
        """화면 정중앙에서 가로·세로가 scale 비율인 네모를 돌려줍니다."""
        h, w = frame.shape[:2]
        rw = int(w * scale)
        rh = int(h * scale)
        x1 = (w - rw) // 2
        y1 = (h - rh) // 2
        return x1, y1, rw, rh

    def top_box(self, frame, scale):
        """가로는 가운데, 세로는 화면 위에 붙인 네모를 돌려줍니다."""
        x1, _y1, rw, rh = self.center_box(frame, scale)
        return x1, 0, rw, rh

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
    # 손 크기는 손목에서 중지 뿌리까지로 나눈다.
    # 지금 학습은 여기에 엄지, 검지, 중지 끝의 z, dz, 손가락마다 굽은 각,
    # 그리고 중지 10번 관절의 x, y, z 를 더한다. 좌클릭은 엄지가 이 관절에 닿는다.
    FEATURE_KIND = "fingertips_joint10"
    WINDOW_KINDS = (
        "fingertips_joint10",
        "fingertips_angle",
        "fingertips_z",
        "fingertips",
        "mediapipe_joints",
        "mediapipe",
    )
    # (각을 재는 관절, 한쪽, 다른 쪽). 화면의 가로·세로만 쓴다.
    _BENDS = ((3, 2, 4), (6, 5, 8), (10, 9, 12), (14, 13, 16), (18, 17, 20))
    _TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
    _TIPS = (4, 8, 12, 16, 20)
    _Z_TIPS = (4, 8, 12)
    MOTION_FEATURES = tuple(
        f"{name}_{axis}"
        for name in _TIP_NAMES
        for axis in ("x", "y", "dx", "dy")
    )
    LANDMARK_COUNT = 21
    _WRIST = 0
    _MIDDLE_MCP = 9
    _MIDDLE_PIP = 10

    def close(self):
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def window_features(self, frames):
        """1초 묶음을 다섯 손가락 끝의 위치와 이동으로 바꿉니다. 손이 거의 없으면 None입니다."""
        valid, scale = self._valid_frames(frames)
        if valid is None:
            return None
        return self._tip_rows_joint10(valid, scale)

    def augmented_features(self, frames):
        """학습용. 원본과 사다리꼴 네 벌에, 회전·속도·흔들림을 각각 붙입니다."""
        valid, scale = self._valid_frames(frames)
        if valid is None:
            return []
        rows = self._augment_rows(valid, scale)
        for amount in self.settings.augment_trapezoids:
            shaped = self._trapezoid_frames(valid, float(amount))
            rows.extend(self._augment_rows(shaped, self._frame_scale(shaped)))
        return rows

    SEQUENCE_LENGTH = 15

    def sequence_for(self, frames):
        """프레임마다 손 좌표를 시간 순서로 쌓습니다. 길이는 15장으로 맞춥니다."""
        valid, _scale = self._valid_frames(frames)
        if valid is None:
            return None
        return self._sequence_from_valid(valid)

    def augmented_sequences(self, frames):
        """학습용. 원본과 사다리꼴에 회전·속도·흔들림을 붙인 좌표 시계열입니다."""
        valid, scale = self._valid_frames(frames)
        if valid is None:
            return []
        rows = self._sequence_copies(valid, scale)
        for amount in self.settings.augment_trapezoids:
            shaped = self._trapezoid_frames(valid, float(amount))
            rows.extend(self._sequence_copies(shaped, self._frame_scale(shaped)))
        return rows

    def _sequence_copies(self, valid, scale):
        """한 벌의 좌표에서 원본, 회전, 속도, 흔들림 시계열을 만듭니다."""
        rows = [self._sequence_from_valid(valid)]
        for degrees in self.settings.augment_rotations:
            rows.append(self._sequence_from_valid(self._rotate_frames(valid, degrees)))
        for speed in self.settings.augment_speeds:
            rows.append(self._sequence_from_valid(self._warp_frames(valid, speed)))
        rng = np.random.default_rng(self._augment_seed(valid))
        for _copy in range(self.settings.augment_jitter_copies):
            rows.append(self._sequence_from_valid(self._jitter_frames(valid, scale, rng)))
        return rows

    def _sequence_from_valid(self, valid):
        """손 좌표 묶음을 15장 길이의 시계열로 맞춥니다."""
        items = self._resample_valid(valid, self.SEQUENCE_LENGTH)
        return np.stack([self._frame_vector(item) for item in items]).astype(np.float32)

    def _resample_valid(self, valid, length):
        """장 수가 달라도 같은 길이로 고릅니다."""
        if len(valid) == length:
            return valid
        indexes = np.linspace(0, len(valid) - 1, num=length)
        last = len(valid) - 1
        return [valid[min(int(round(float(index))), last)] for index in indexes]

    def _frame_vector(self, item):
        """한 장의 관절을 손목 기준 좌표와 굽은 각으로 바꿉니다."""
        scale = float(item["scale"]) + 1e-6
        xy = (item["joints"] - item["wrist"]) / scale
        depth = (item["depth"] - item["depth"][self._WRIST]) / scale
        angles = np.array(
            [self._bend_degrees(item["joints"], joint, left, right) / 180.0 for joint, left, right in self._BENDS],
            dtype=np.float32,
        )
        return np.concatenate([xy.reshape(-1), depth.astype(np.float32), angles]).astype(np.float32)

    def _augment_rows(self, valid, scale):
        """한 벌의 좌표에서 원본, 회전, 속도, 흔들림 줄을 만듭니다."""
        rows = [self._tip_rows_joint10(valid, scale)]
        for degrees in self.settings.augment_rotations:
            rows.append(self._tip_rows_joint10(self._rotate_frames(valid, degrees), scale))
        for speed in self.settings.augment_speeds:
            warped = self._warp_frames(valid, speed)
            rows.append(self._tip_rows_joint10(warped, self._frame_scale(warped)))
        rng = np.random.default_rng(self._augment_seed(valid))
        for _copy in range(self.settings.augment_jitter_copies):
            shaken = self._jitter_frames(valid, scale, rng)
            rows.append(self._tip_rows_joint10(shaken, self._frame_scale(shaken)))
        return rows

    def _trapezoid_frames(self, valid, amount):
        """가로만 사다리꼴로 바꿉니다. 양수는 위가 넓고, 음수는 아래가 넓습니다."""
        shaped = []
        for item in valid:
            joints = item["joints"].copy()
            # y 는 0 이 화면 위, 1 이 화면 아래다. 가로 배율은 그 사이에서 직선으로 변한다.
            scale_x = (1.0 + amount) - (2.0 * amount) * joints[:, 1]
            joints[:, 0] = 0.5 + (joints[:, 0] - 0.5) * scale_x
            wrist = joints[self._WRIST]
            frame_scale = float(np.linalg.norm(joints[self._MIDDLE_MCP] - wrist))
            if frame_scale < 1e-4:
                frame_scale = float(item["scale"])
            shaped.append(
                {
                    "joints": joints.astype(np.float32),
                    "wrist": wrist,
                    "scale": frame_scale,
                    "depth": item["depth"],
                }
            )
        return shaped

    # 화면 위 위치는 다섯 손가락 끝의 x, y 다. 깊이는 엄지, 검지, 중지 끝만 더한다.
    # z 는 손목을 뺀 뒤 손 크기로 나눈다. 1초 중간의 깊이와, 처음에서 끝까지 깊이가 변한 양이다.
    # 약지와 새끼는 클릭을 구분하는 데 덜 쓰여서 넣지 않는다.
    def _tip_rows_z(self, valid, scale):
        """다섯 손가락 끝의 평면 좌표에 엄지·검지·중지 깊이를 붙입니다."""
        rows = [self._tip_rows(valid, scale)]
        for tip in self._Z_TIPS:
            placed = np.array(
                [(item["depth"][tip] - item["depth"][self._WRIST]) / scale for item in valid],
                dtype=np.float32,
            )
            rows.append(np.array([np.median(placed), placed[-1] - placed[0]], dtype=np.float32))
        return np.concatenate(rows).astype(np.float32)

    # 손가락이 접힌 정도다. 편 손은 180도에 가깝고, 접으면 작아진다.
    # 1초 중간의 각과, 처음에서 끝까지 각이 변한 양을 붙인다.
    def _tip_rows_angle(self, valid, scale):
        """위치·깊이에 다섯 손가락의 굽은 각을 붙입니다."""
        rows = [self._tip_rows_z(valid, scale)]
        for joint, left, right in self._BENDS:
            angles = np.array(
                [self._bend_degrees(item["joints"], joint, left, right) for item in valid],
                dtype=np.float32,
            )
            rows.append(np.array([np.median(angles), angles[-1] - angles[0]], dtype=np.float32))
        return np.concatenate(rows).astype(np.float32)

    # 좌클릭은 엄지 끝이 중지 10번에 닿는다. 끝 좌표만으로는 그 관절 위치가 안 보인다.
    # 다섯 끝과 같이, 1초 중간의 x, y, z 와 그 사이 움직인 양을 붙인다.
    def _tip_rows_joint10(self, valid, scale):
        """굽은 각 뒤에 중지 10번 관절의 위치와 이동을 붙입니다."""
        return np.concatenate(
            [self._tip_rows_angle(valid, scale), self._point_delta(valid, scale, self._MIDDLE_PIP)]
        ).astype(np.float32)

    def _point_delta(self, valid, scale, index):
        """한 관절의 중간 x, y, z 와 처음에서 끝까지의 이동."""
        placed = np.stack([(item["joints"][index] - item["wrist"]) / scale for item in valid])
        xy = np.concatenate([np.median(placed, axis=0), placed[-1] - placed[0]])
        depth = np.array(
            [(item["depth"][index] - item["depth"][self._WRIST]) / scale for item in valid],
            dtype=np.float32,
        )
        z = np.array([np.median(depth), depth[-1] - depth[0]], dtype=np.float32)
        return np.concatenate([xy, z]).astype(np.float32)

    def _bend_degrees(self, joints, joint, left, right):
        """관절 joint 에서 left, right 가 이루는 각. 점은 거의 겹치면 편 손으로 봅니다."""
        before = joints[left] - joints[joint]
        after = joints[right] - joints[joint]
        length_before = float(np.linalg.norm(before))
        length_after = float(np.linalg.norm(after))
        if length_before < 1e-6 or length_after < 1e-6:
            return 180.0
        cosine = float(np.dot(before, after) / (length_before * length_after))
        cosine = min(1.0, max(-1.0, cosine))
        return float(np.degrees(np.arccos(cosine)))

    def _tip_rows(self, valid, scale):
        """손목 기준 다섯 손가락 끝의 중간 위치와 처음에서 끝까지의 이동."""
        rows = []
        for tip in self._TIPS:
            placed = np.stack([(item["joints"][tip] - item["wrist"]) / scale for item in valid])
            rows.append(np.concatenate([np.median(placed, axis=0), placed[-1] - placed[0]]))
        return np.concatenate(rows).astype(np.float32)

    def _rotate_frames(self, valid, degrees):
        """손목을 중심으로 손 좌표를 돌립니다. 화면에서의 위치는 그대로입니다."""
        theta = np.deg2rad(degrees)
        cosine, sine = float(np.cos(theta)), float(np.sin(theta))
        rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float32)
        rotated = []
        for item in valid:
            joints = (item["joints"] - item["wrist"]) @ rotation + item["wrist"]
            rotated.append(
                {"joints": joints, "wrist": item["wrist"], "scale": item["scale"], "depth": item["depth"]}
            )
        return rotated

    def _warp_frames(self, valid, speed):
        """같은 동작을 더 빠르거나 느리게 한 좌표. 1보다 크면 일찍 끝나고 그 자세로 남습니다."""
        count = len(valid)
        if count < 2:
            return valid
        positions = np.clip(np.arange(count, dtype=np.float64) * float(speed), 0, count - 1)
        warped = []
        for position in positions:
            left = int(np.floor(position))
            right = min(left + 1, count - 1)
            weight = float(position - left)
            joints = ((1.0 - weight) * valid[left]["joints"] + weight * valid[right]["joints"]).astype(np.float32)
            depth = ((1.0 - weight) * valid[left]["depth"] + weight * valid[right]["depth"]).astype(np.float32)
            wrist = joints[self._WRIST]
            scale = float(np.linalg.norm(joints[self._MIDDLE_MCP] - wrist))
            if scale < 1e-4:
                scale = float(valid[left]["scale"])
            warped.append({"joints": joints, "wrist": wrist, "scale": scale, "depth": depth})
        return warped

    def _jitter_frames(self, valid, scale, rng):
        """관절 좌표에 손 크기 대비 작은 흔들림을 더합니다."""
        sigma = float(self.settings.augment_jitter) * float(scale)
        shaken = []
        for item in valid:
            joints = item["joints"] + rng.normal(0.0, sigma, size=item["joints"].shape).astype(np.float32)
            depth = item["depth"] + rng.normal(0.0, sigma, size=item["depth"].shape).astype(np.float32)
            wrist = joints[self._WRIST]
            frame_scale = float(np.linalg.norm(joints[self._MIDDLE_MCP] - wrist))
            if frame_scale < 1e-4:
                frame_scale = float(item["scale"])
            shaken.append({"joints": joints, "wrist": wrist, "scale": frame_scale, "depth": depth})
        return shaken

    def _frame_scale(self, valid):
        return float(np.median([item["scale"] for item in valid])) + 1e-6

    def _augment_seed(self, valid):
        tips = valid[0]["joints"][list(self._TIPS)] - valid[0]["wrist"]
        return int(np.abs(tips).sum() * 1_000_000) % (2**32)

    def _joint_features(self, frames):
        """예전 모델용. 관절 21개의 위치와 화면 이동을 한 줄로 만듭니다."""
        valid, scale = self._valid_frames(frames)
        if valid is None:
            return None
        placed = [(item["joints"] - item["wrist"]) / scale for item in valid]
        shape = np.median(np.stack(placed), axis=0)
        motion = (valid[-1]["joints"] - valid[0]["joints"]) / scale
        return np.concatenate([shape, motion], axis=1).reshape(-1).astype(np.float32)

    # 인식 화면에 적는 값이다. 학습과 같이 손목을 빼고 손 크기로 나눈다.
    # 다섯 끝은 x, y 다. z 는 엄지, 검지, 중지만 있다.
    def tip_coord_labels(self, frame):
        """학습에 쓰는 손가락 끝의 x, y, z 글자. 손이 없으면 None입니다."""
        item = self._finger_points(frame)
        if item is None:
            return None
        scale = float(item["scale"]) + 1e-6
        wrist_z = float(item["depth"][self._WRIST])
        labels = []
        for name, tip in zip(self._TIP_NAMES, self._TIPS):
            xy = (item["joints"][tip] - item["wrist"]) / scale
            text = f"{tip} {name} x{xy[0]:+.2f} y{xy[1]:+.2f}"
            if tip in self._Z_TIPS:
                z = (float(item["depth"][tip]) - wrist_z) / scale
                text += f" z{z:+.2f}"
            if tip == 12 and xy[0] < self.settings.dorsum_x:
                text += " dorsum"
            elif tip == 12 and xy[0] > self.settings.palm_x:
                text += " palm"
            labels.append(text)
        pip = (item["joints"][self._MIDDLE_PIP] - item["wrist"]) / scale
        pip_z = (float(item["depth"][self._MIDDLE_PIP]) - wrist_z) / scale
        labels.append(f"10 pip x{pip[0]:+.2f} y{pip[1]:+.2f} z{pip_z:+.2f}")
        return labels

    def middle_x(self, frame):
        """오른손 중지 끝의 가로 위치. 손목을 빼고 손 크기로 나눈 값입니다."""
        item = self._finger_points(frame)
        if item is None:
            return None
        scale = float(item["scale"]) + 1e-6
        placed = (item["joints"][12] - item["wrist"]) / scale
        return float(placed[0])

    # 커서는 검지 끝을 따라가다가, 클릭하려고 검지가 내려가면 그 끝은 더 이상 보면 안 된다.
    # 내려가기 직전의 검지 끝과 그 순간의 손목을 같이 기억하려고 둘을 한 번에 꺼낸다.
    # 손목만 비교하면 손은 옮겨도, 검지를 구부린 움직임은 커서에 섞이지 않는다.
    # 값은 화면 왼쪽 위가 0인 픽셀이다. 손이 없으면 None이다.
    def pointing(self, frame):
        """검지 끝과 손목의 화면 픽셀. 손이 없으면 None입니다."""
        item = self._finger_points(frame)
        if item is None:
            return None
        height, width = frame.shape[:2]
        tip = item["joints"][8]
        wrist = item["wrist"]
        return (
            float(tip[0]) * width,
            float(tip[1]) * height,
            float(wrist[0]) * width,
            float(wrist[1]) * height,
        )

    def contacts(self, frame):
        """지금 손가락 끝이 엄지와 얼마나 가까운지. 손이 없으면 None입니다.

        좌클릭만 엄지 끝과 중지 10번 거리를 본다. 우클릭은 모델 판정만 쓴다.
        """
        item = self._finger_points(frame)
        if item is None:
            return None
        joints = item["joints"]
        scale = float(item["scale"]) + 1e-6
        thumb = joints[4]

        def gap(index):
            return float(np.linalg.norm(joints[index] - thumb) / scale)

        return {
            "left_click": gap(self._MIDDLE_PIP),
        }

    def tip_motion(self, frames):
        """손가락 끝이 손목 기준으로 움직인 거리. 손이 없으면 None, 한 장이면 0입니다."""
        points = [self._finger_points(frame) for frame in frames]
        valid = [item for item in points if item is not None]
        if not valid:
            return None
        if len(valid) < 2:
            return 0.0
        scale = float(np.median([item["scale"] for item in valid])) + 1e-6
        moved = 0.0
        for tip in self._TIPS:
            start = (valid[0]["joints"][tip] - valid[0]["wrist"]) / scale
            end = (valid[-1]["joints"][tip] - valid[-1]["wrist"]) / scale
            moved = max(moved, float(np.linalg.norm(end - start)))
        return moved

    def _valid_frames(self, frames):
        """손이 잡힌 장면과 손 크기를 고릅니다. 너무 적으면 (None, None)입니다."""
        points = [self._finger_points(frame) for frame in frames]
        valid = [item for item in points if item is not None]
        # 1초가 안 되는 영상은 손이 한 장이라도 있으면 넣는다.
        need = 1 if len(frames) < 8 else max(3, len(frames) // 2)
        if len(valid) < need:
            return None, None
        scale = float(np.median([item["scale"] for item in valid])) + 1e-6
        return valid, scale

    def features_for(self, frames, kind):
        """고른 모델이 학습할 때 본 것과 같은 숫자 한 줄을 만듭니다."""
        # 위치, 깊이, 굽은 각, 중지 10번. 지금 학습 버튼이 만든다.
        if kind == self.FEATURE_KIND:
            return self.window_features(frames)
        # 10번 없는 모델. 14번부터 17번까지다.
        if kind == "fingertips_angle":
            valid, scale = self._valid_frames(frames)
            if valid is None:
                return None
            return self._tip_rows_angle(valid, scale)
        # 각 없는 다섯 손가락 끝과 엄지·검지·중지 깊이. 모델 9부터 11까지다.
        if kind == "fingertips_z":
            valid, scale = self._valid_frames(frames)
            if valid is None:
                return None
            return self._tip_rows_z(valid, scale)
        # 깊이 없는 다섯 손가락 끝. 이전에 학습한 모델이다.
        if kind == "fingertips":
            valid, scale = self._valid_frames(frames)
            if valid is None:
                return None
            return self._tip_rows(valid, scale)
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

    def landmarks(self, frame):
        """이 화면의 관절입니다. 같은 화면이면 미디어파이프를 다시 돌리지 않습니다."""
        return self._finger_points(frame)

    def _finger_points(self, bgr):
        """한 장에서 관절 21개를 찾습니다. 손이 없으면 None입니다."""
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
        # 화면 가로·세로에 대한 0~1 좌표. z 는 화면 가로 기준의 깊이이고, 학습에서 엄지·검지·중지만 쓴다.
        joints = np.array([[point.x, point.y] for point in landmark], dtype=np.float32)
        depth = np.array([point.z for point in landmark], dtype=np.float32)
        if len(joints) != self.LANDMARK_COUNT:
            return None
        wrist = joints[self._WRIST]
        scale = float(np.linalg.norm(joints[self._MIDDLE_MCP] - wrist))
        if scale < 1e-4:
            return None
        return {"joints": joints, "wrist": wrist, "scale": scale, "depth": depth}

    def use_static(self, static):
        """학습 영상은 장마다 따로 찾고, 카메라는 이전 손을 이어 추적합니다."""
        self._static_image = bool(static)

    def _ensure_hands(self):
        """손 모델을 엽니다. 카메라와 학습의 찾기 방식이 다르면 다시 엽니다."""
        static = self._static_image
        if self._hands is not None and self._hands_static == static:
            return self._hands
        if self.hands_error and self._hands is None and self._hands_static == static:
            return None
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        try:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=static,
                max_num_hands=1,
                model_complexity=self.settings.hand_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as exc:
            self.hands_error = f"손 좌표 모델을 열지 못했습니다. {exc}"
            self._hands_static = static
            return None
        self.hands_error = None
        self._hands_static = static
        return self._hands
