"""웹캠 프레임을 보여주고, 녹화 중이면 중앙 영역만 저장합니다."""

import cv2
import numpy as np

from gesture.recorder import open_camera


class CameraPane:
    """인식 탭과 등록 탭이 같은 카메라 루프를 씁니다."""

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

    # 학습 영상 2. 좌우를 뒤집은 화면에서 중앙만 자른다.
    # 미리보기는 전체 화면에 박스를 그리고, 녹화 중이면 박스 안만 파일에 넣는다.
    def _show(self, frame):
        roi, box = self.analyzer.crop(frame)
        label, conf = self._predict(roi)
        if self.model is None and label == "none":
            text = "제스처: 모델 없음"
        else:
            name = self.settings.gesture_ko.get(label, self._title(label))
            text = f"제스처: {name}  {round(conf * 100)}%"
        self.recognize_status.setText(text)

        view = frame.copy()
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

    def _predict(self, roi):
        feat, _mask = self.analyzer.extract(roi)
        if feat is None or self.model is None:
            self.smoother.reset()
            return "none", 0.0
        proba = self.model.predict_proba(feat.reshape(1, -1))[0]
        ordered = np.zeros(len(self.model_tags), dtype=np.float64)
        for index, cls in enumerate(self.model.classes_):
            ordered[int(cls)] = proba[index]
        label, conf, _action, _fired = self.smoother.update(ordered)
        return label, conf
