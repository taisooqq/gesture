"""웹캠 프레임을 분류해 슬라이드 명령으로 바꿉니다."""

import time

import cv2
import joblib
import numpy as np

from gesture.config import settings as default_settings
from gesture.hand import HandAnalyzer
from gesture.recorder import open_camera
from gesture.slides import SlideWindow
from gesture.smoother import GestureSmoother


class LiveDemo:
    """프레임 → 손 특징 → 모델 → 스무딩 → 슬라이드 창."""

    def __init__(self, camera=0, ui_only=False, settings=None, analyzer=None):
        self.camera = camera
        self.ui_only = ui_only
        self.settings = settings or default_settings
        self.analyzer = analyzer or HandAnalyzer(self.settings)
        self.slides = SlideWindow(self.settings)
        self.smoother = GestureSmoother(self.settings)
        self.model = None

    def run(self):
        self.slides.start()
        if self.ui_only:
            self.slides.wait()
            return
        self.model = self._load_model()
        self._camera_loop()

    def _load_model(self):
        path = self.settings.model_path
        if not path.exists():
            print("모델 없음. 마스크만 표시합니다. collect 다음 train 을 실행하세요.")
            return None
        blob = joblib.load(path)
        print("모델 로드:", path)
        return blob["pipeline"]

    def _camera_loop(self):
        cap = open_camera(self.camera)
        if not cap.isOpened():
            self.slides.shutdown()
            raise SystemExit("웹캠을 열 수 없습니다.")

        fps = 0.0
        prev = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.perf_counter()
                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                if not self._step(cv2.flip(frame, 1), fps):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.slides.shutdown()

    def _step(self, frame, fps):
        roi, (x, y, w, h) = self.analyzer.crop(frame)
        feat, mask = self.analyzer.extract(roi)
        label, conf, action, fired = self._predict(feat)
        self.slides.publish(label, conf, fps, action if fired else None)
        self._draw(frame, roi, mask, (x, y, w, h), label, conf, action if fired else "none", fps)
        if self.slides.closed:
            return False
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)

    def _predict(self, feat):
        if feat is None or self.model is None:
            self.smoother.reset()
            return "none", 0.0, "none", False
        proba = self.model.predict_proba(feat.reshape(1, -1))[0]
        ordered = np.zeros(len(self.settings.gestures), dtype=np.float64)
        for i, cls in enumerate(self.model.classes_):
            ordered[int(cls)] = proba[i]
        return self.smoother.update(ordered)

    def _draw(self, frame, roi, mask, box, label, conf, action, fps):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
        text = f"{label} {conf:.2f}  {action}  {fps:.0f} fps"
        cv2.putText(frame, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        view = np.hstack([roi, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow("demo", frame)
        cv2.imshow("roi | mask", view)
