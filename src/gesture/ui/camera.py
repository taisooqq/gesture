"""웹캠 프레임을 보여주고, 녹화 중이면 중앙 영역만 저장합니다."""

import time
from collections import deque

import cv2
import numpy as np

from gesture.hand import HandAnalyzer
from gesture.hand_track import HandOverlay
from gesture.recorder import VideoRecorder, open_camera
from gesture.smoother import GestureSmoother


class CameraPane:
    """인식 탭과 등록 탭이 같은 카메라 루프를 씁니다."""

    analyzer: HandAnalyzer
    hand_overlay: HandOverlay
    recorder: VideoRecorder
    smoother: GestureSmoother

    def _open_camera(self):
        self.cap = open_camera(self.camera_index)
        if not self.cap.isOpened():
            self.recognize_status.setText("웹캠을 열 수 없습니다.")
            self.register_status.setText("웹캠을 열 수 없습니다.")
            self.cap.release()
            self.cap = None

    def _tick(self):
        if self._closing:
            return
        if self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                self._show(cv2.flip(frame, 1))
        self._play_clip()

    # 인식은 화면 전체를 본다. 중앙 박스는 등록 탭에서 녹화할 영역만 표시한다.
    def _show(self, frame):
        roi, box = self.analyzer.crop(frame)
        label, conf = self._predict(frame)
        if getattr(self, "_predict_note", None):
            text = self._predict_note
        elif self.model is None and label == "none":
            text = "제스처: 모델 없음"
        else:
            name = self.settings.gesture_ko.get(label, self._title(label))
            if label in ("none", "idle"):
                text = f"제스처: {name}"
            else:
                text = f"제스처: {name}  {round(conf * 100)}%"
        self.recognize_status.setText(text)

        view = self.hand_overlay.draw(frame)
        if self.hand_overlay.error:
            self.register_status.setText(self.hand_overlay.error)
        if self.tabs.currentIndex() != 0:
            x, y, w, h = box
            color = (40, 40, 220) if self.recorder.recording else (60, 190, 80)
            cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        target = self.recognize_image if self.tabs.currentIndex() == 0 else self.register_image
        self._set_image(target, view)

        if not self.recorder.recording:
            return
        try:
            self.recorder.write(roi)
        except RuntimeError as exc:
            self.recorder.recording = False
            self.record_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.register_status.setText(str(exc))
            return
        self.register_status.setText(
            f"녹화 중 · {self._title(self.tag)} · {self.recorder.elapsed():.1f}초"
        )

    # 인식. 고른 모델이 학습한 숫자 종류로 최근 화면을 바꿔 그 모델에 넣는다.
    def _predict(self, frame):
        self._predict_note = None
        if not hasattr(self, "_recent_frames"):
            self._recent_frames = deque()
        now = time.perf_counter()
        self._recent_frames.append((now, frame))
        while self._recent_frames and now - self._recent_frames[0][0] > 1.0:
            self._recent_frames.popleft()
        if self.model is None:
            self._phase = None
            self.smoother.reset()
            return "none", 0.0
        if not hasattr(self, "_phase"):
            self._phase = None
        # 손이 없으면 동작도 없다. 무동작은 학습 항목이 아니라, 잡고 있던 손을 놓은 상태다.
        contact = self.analyzer.contacts(frame)
        if contact is None:
            self._phase = None
            self.smoother.reset()
            return "none", 0.0
        # 좌클릭·우클릭은 손가락이 맞닿아 있는 동안 유지한다. 드래그는 이 상태가 끝나기 전이다.
        if self._phase in self.settings.hold_tags:
            if contact[self._phase] < self.settings.pinch_off:
                return self._phase, getattr(self, "_phase_conf", 1.0)
            self._phase = None
            self.smoother.reset()
            return "idle", 0.0
        kind = getattr(self, "model_features", None)
        # 윤곽 숫자 8개 모델은 한 장만 본다. 관절 모델은 최근 1초가 모여야 한다.
        single = kind not in HandAnalyzer.WINDOW_KINDS
        if not single and now - self._recent_frames[0][0] < 0.8:
            return "idle", 0.0
        frames = [frame] if single else self._even_frames(15)
        feat = self.analyzer.features_for(frames, kind)
        expected = getattr(self.model, "n_features_in_", None)
        if feat is None or (expected is not None and feat.shape[-1] != expected):
            self.smoother.reset()
            return "idle", 0.0
        proba = self.model.predict_proba(feat.reshape(1, -1))[0]
        ordered = np.zeros(len(self.model_tags), dtype=np.float64)
        for index, cls in enumerate(self.model.classes_):
            ordered[int(cls)] = proba[index]
        label, conf, _action, _fired = self.smoother.update(ordered)
        # 맞닿아야 시작하는 동작. 모델이 말해도 손가락이 떨어져 있으면 시작하지 않는다.
        if label in self.settings.hold_tags and contact[label] < self.settings.pinch_on:
            self._phase = label
            self._phase_conf = conf
            return label, conf
        # 스와이프·원처럼 한 번 하는 동작. 움직임이 끝나면 끝난 자세를 학습하지 않고 무동작이다.
        if label not in self.settings.hold_tags and label not in ("none", "idle"):
            motion = self.analyzer.tip_motion(self._frames_within(self.settings.idle_seconds))
            if motion is not None and motion >= self.settings.idle_motion:
                return label, conf
        return "idle", 0.0

    def _frames_within(self, seconds):
        """최근 몇 초 안의 화면만 고릅니다."""
        if not self._recent_frames:
            return []
        now = self._recent_frames[-1][0]
        return [frame for stamp, frame in self._recent_frames if now - stamp <= seconds]

    def _even_frames(self, count):
        """최근 1초에 모인 화면을 학습 묶음과 같은 장 수로 고릅니다."""
        frames = [frame for _stamp, frame in self._recent_frames]
        if len(frames) <= count:
            return frames
        indexes = np.linspace(0, len(frames) - 1, num=count, dtype=int)
        return [frames[int(index)] for index in indexes]
