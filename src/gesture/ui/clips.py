"""저장 영상 목록, 재생, 삭제와 녹화 시작·종료."""

import time
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem, QMessageBox

from gesture.recorder import open_video


class ClipPane:
    """태그 폴더의 avi를 고르고, 찍힌 속도로 재생합니다."""

    def _toggle_record(self):
        if self.recorder.recording:
            self._stop_record()
        else:
            self._start_record()

    # 학습 영상 1. 태그가 있을 때만 녹화를 연다. 파일은 아직 만들지 않는다.
    def _start_record(self):
        if self.recorder.recording:
            return
        if not self.tag:
            self.register_status.setText("오른쪽에서 태그를 먼저 고르세요.")
            return
        self.recorder.start(self.tag)
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.register_status.setText(f"녹화 중 · {self._title(self.tag)}")

    # 학습 영상 5. 녹화를 닫고, 방금 파일은 선택한 태그의 목록에 붙인다.
    def _stop_record(self):
        if not self.recorder.recording:
            return
        try:
            path = self.recorder.stop()
        except cv2.error:
            path = None
        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if path is None:
            self.register_status.setText("저장된 프레임이 없습니다.")
        else:
            self.register_status.setText(f"저장했습니다. {path.name}")
            self._refresh_tag(self.tag)
            self._reload_clips(select=path)

    def _reload_clips(self, select=None):
        if not hasattr(self, "clip_list"):
            return
        self.clip_header.setText(self._title(self.tag) if self.tag else "영상")
        selected = Path(select) if select else self.clip_path
        self.clip_list.blockSignals(True)
        self.clip_list.clear()
        trained = self.model_store.trained_videos()
        paths = []
        if self.tag:
            folder = self.settings.video_dir / self.tag
            if folder.exists():
                paths = sorted(
                    folder.glob("*.avi"),
                    key=lambda path: (
                        path.relative_to(self.settings.video_dir).as_posix() in trained,
                        path.name,
                    ),
                )
        chosen_row = -1
        for path in paths:
            relative = path.relative_to(self.settings.video_dir).as_posix()
            mark = "학습됨" if relative in trained else "미학습"
            item = QListWidgetItem(f"{path.name}    {mark}")
            item.setData(Qt.UserRole, str(path))
            self.clip_list.addItem(item)
            if selected is not None and path == Path(selected):
                chosen_row = self.clip_list.count() - 1
        self.clip_list.blockSignals(False)
        if chosen_row >= 0:
            self.clip_list.setCurrentRow(chosen_row)
        elif self.clip_list.count():
            self.clip_list.setCurrentRow(0)
        else:
            self._close_clip()
            self._mark_selected()

    def _select_clip(self, current, _previous):
        if current is None:
            self._close_clip()
            self._mark_selected()
            return
        self._open_clip(Path(current.data(Qt.UserRole)))

    def _open_clip(self, path):
        if self.clip_path == path and self.clip_cap is not None:
            return
        self._close_clip()
        cap = open_video(path)
        if not cap.isOpened():
            self.register_status.setText("영상을 열 수 없습니다.")
            self._mark_selected()
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.clip_fps = fps if 1 <= fps <= 60 else 15.0
        self.clip_next = 0.0
        self.clip_cap = cap
        self.clip_path = path
        self._mark_selected()

    def _close_clip(self):
        if self.clip_cap is not None:
            self.clip_cap.release()
        self.clip_cap = None
        self.clip_path = None
        if hasattr(self, "clip_image"):
            self.clip_image.clear()

    def _play_clip(self):
        if self.clip_cap is None:
            return
        now = time.perf_counter()
        if now < self.clip_next:
            return
        self.clip_next = now + 1.0 / self.clip_fps
        ok, frame = self.clip_cap.read()
        if not ok:
            self.clip_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.clip_cap.read()
        if ok:
            self._set_image(self.clip_image, frame)

    def _delete_clip(self):
        if self.recorder.recording or self.clip_path is None:
            return
        path = self.clip_path
        relative = path.relative_to(self.settings.video_dir).as_posix()
        trained = relative in self.model_store.trained_videos()
        note = "\n이 영상은 이미 학습에 사용되었습니다." if trained else ""
        answer = QMessageBox.question(
            self.window,
            "영상 삭제",
            f"{path.name} 파일을 삭제할까요?{note}",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._close_clip()
        if path.exists():
            path.unlink()
        self._reload_clips()
        if self.tag:
            self._refresh_tag(self.tag)
        self.register_status.setText("영상을 삭제했습니다.")
