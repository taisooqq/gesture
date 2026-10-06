"""캠 화면에 미디어파이프 손 좌표를 그립니다.

예제의 mp.solutions.hands 를 그대로 씁니다.
미디어파이프 1.0 에는 이 API가 없고, 0.10.18 에서 동작합니다.
"""

import cv2
import mediapipe as mp


class HandOverlay:
    """한 손의 랜드마크와 연결선을 카메라 화면 위에 그립니다."""

    def __init__(self):
        self._hands = None
        self._drawing = mp.solutions.drawing_utils
        self._hands_api = mp.solutions.hands
        self.error = None

    def close(self):
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def draw(self, bgr):
        """좌우가 뒤집힌 BGR 화면을 받아, 손 좌표를 그린 화면을 돌려줍니다."""
        if self.error:
            return bgr
        try:
            self._ensure()
        except Exception as exc:
            self.error = f"손 좌표 모델을 열지 못했습니다. {exc}"
            return bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return bgr
        view = bgr.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            self._drawing.draw_landmarks(
                view, hand_landmarks, self._hands_api.HAND_CONNECTIONS
            )
            self._write_coordinates(view, hand_landmarks)
        return view

    def _write_coordinates(self, view, hand_landmarks):
        """관절 번호는 점 옆에, 픽셀 좌표는 화면 오른쪽에 적습니다."""
        height, width = view.shape[:2]
        scale = max(height / 1200, 1)
        lines = []
        for index, point in enumerate(hand_landmarks.landmark):
            x = min(max(int(point.x * width), 0), width - 1)
            y = min(max(int(point.y * height), 0), height - 1)
            self._text(view, str(index), (x + 6, y - 6), scale)
            lines.append(f"{index:2d}  {x:4d},{y:4d}")
        x0 = width - int(150 * scale)
        y0 = int(22 * scale)
        for row, line in enumerate(lines):
            self._text(view, line, (x0, y0 + int(row * 18 * scale)), scale)

    def _text(self, view, text, origin, scale):
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.45 * scale
        cv2.putText(view, text, origin, font, size, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, origin, font, size, (255, 255, 255), 1, cv2.LINE_AA)

    def _ensure(self):
        if self._hands is not None:
            return
        self._hands = self._hands_api.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
