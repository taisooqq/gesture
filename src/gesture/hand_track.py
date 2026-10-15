"""캠 화면에 이미 찾은 손 좌표를 그립니다.

관절은 HandAnalyzer 가 한 번 찾은 값을 그대로 씁니다.
"""

import cv2
import mediapipe as mp


class HandOverlay:
    """한 손의 랜드마크와 연결선을 카메라 화면 위에 그립니다."""

    def __init__(self):
        self._connections = mp.solutions.hands.HAND_CONNECTIONS
        self.error = None

    def close(self):
        return None

    def draw(self, bgr, points, numbers=True):
        """이미 찾은 관절로 손 선을 그립니다. 미디어파이프를 다시 돌리지 않습니다."""
        if not points:
            return bgr
        joints = points["joints"]
        height, width = bgr.shape[:2]
        view = bgr.copy()
        for start, end in self._connections:
            cv2.line(
                view,
                self._pixel(joints, start, width, height),
                self._pixel(joints, end, width, height),
                (80, 220, 60),
                2,
            )
        for index in range(len(joints)):
            origin = self._pixel(joints, index, width, height)
            cv2.circle(view, origin, 3, (40, 40, 220), -1)
            if numbers:
                self._text(view, str(index), (origin[0] + 6, origin[1] - 6), max(height / 1200, 3))
        return view

    def _pixel(self, joints, index, width, height):
        point = joints[index]
        x = min(max(int(point[0] * width), 0), width - 1)
        y = min(max(int(point[1] * height), 0), height - 1)
        return x, y

    def write_tip_readout(self, view, lines):
        """학습 좌표를 화면 오른쪽 위에 위에서부터 적습니다. 손 위에는 적지 않습니다."""
        if not lines:
            return view
        if view is None:
            return view
        width = view.shape[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.72
        gap = 8
        margin = 12
        y = margin
        for text in lines:
            (tw, th), _ = cv2.getTextSize(text, font, size, 5)
            y += th
            origin = (max(margin, width - tw - margin), y)
            self._text(view, text, origin, 1.6)
            y += gap
        return view

    def _text(self, view, text, origin, scale):
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.45 * scale
        cv2.putText(view, text, origin, font, size, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, origin, font, size, (255, 255, 255), 1, cv2.LINE_AA)
