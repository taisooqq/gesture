"""저장 영상·사진 목록, 재생, 삭제와 녹화·사진 저장."""

import time
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem

from gesture.model.store import ModelStore
from gesture.recorder import VideoRecorder, open_video


class ClipPane:
    """태그 폴더의 avi·사진을 고르고, 찍힌 속도로 재생합니다."""

    model_store: ModelStore
    recorder: VideoRecorder

    def _is_photo_tag(self, tag=None):
        name = self.tag if tag is None else tag
        return name in getattr(self.settings, "photo_tags", ())

    def _toggle_record(self):
        if self._is_photo_tag():
            self._snap_photo()
            return
        if self.recorder.recording:
            self._stop_record()
        else:
            self._start_record()

    def _update_capture_ui(self):
        """무동작 태그는 사진 버튼, 나머지는 녹화·정지."""
        if not hasattr(self, "record_button"):
            return
        photo = self._is_photo_tag()
        self.record_button.setText("사진" if photo else "녹화")
        if photo:
            self.stop_button.setEnabled(False)
        elif not self.recorder.recording:
            self.record_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    # 학습 영상 1. 태그가 있을 때만 녹화를 연다. 파일은 아직 만들지 않는다.
    def _start_record(self):
        if self._is_photo_tag():
            self._snap_photo()
            return
        if self.recorder.recording:
            return
        if not self.tag:
            self.register_status.setText("오른쪽에서 태그를 먼저 고르세요.")
            return
        self.recorder.start(self.tag)
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.register_status.setText(f"녹화 중 · {self._title(self.tag)}")

    def _snap_photo(self):
        """지금 등록 화면 중앙 박스를 jpg로 저장합니다."""
        if self.recorder.recording:
            return
        if not self.tag:
            self.register_status.setText("오른쪽에서 태그를 먼저 고르세요.")
            return
        frame = getattr(self, "_register_roi", None)
        if frame is None:
            self.register_status.setText("카메라 화면이 없습니다.")
            return
        try:
            path = self.recorder.save_photo(self.tag, frame)
        except RuntimeError as exc:
            self.register_status.setText(str(exc))
            return
        self.register_status.setText(f"사진을 저장했습니다. {path.name}")
        self._refresh_tag(self.tag)
        self._reload_clips(select=path)

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
        self._update_capture_ui()
        label = "사진" if self._is_photo_tag() else "영상"
        self.clip_header.setText(self._title(self.tag) if self.tag else label)
        selected = Path(select) if select else self.clip_path
        self.clip_list.blockSignals(True)
        self.clip_list.clear()
        trained = self.model_store.trained_videos()
        paths = self._clip_paths()
        chosen_row = -1
        for path in paths:
            relative = path.relative_to(self.settings.video_dir).as_posix()
            mark = "학습됨" if relative in trained else "미학습"
            if path.suffix.lower() in {item.lower() for item in self.settings.photo_extensions}:
                count = "사진 1장"
            else:
                frames = _frame_count(path)
                count = f"{frames}프레임" if frames else "프레임 없음"
            item = QListWidgetItem(f"{path.name}    {count}    {mark}")
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

    def _clip_paths(self):
        if not self.tag:
            return []
        folder = self.settings.video_dir / self.tag
        if not folder.exists():
            return []
        trained = self.model_store.trained_videos()

        def sort_key(path):
            return (
                path.relative_to(self.settings.video_dir).as_posix() in trained,
                path.name,
            )

        if self._is_photo_tag():
            paths = []
            for extension in self.settings.photo_extensions:
                paths.extend(folder.glob(f"*{extension}"))
            return sorted(paths, key=sort_key)
        return sorted(folder.glob("*.avi"), key=sort_key)

    def _select_clip(self, current, _previous):
        if current is None:
            self._close_clip()
            self._mark_selected()
            return
        self._open_clip(Path(current.data(Qt.UserRole)))

    def _focus_clips(self, _item=None):
        """영상 목록을 누르면 방향키가 영상으로 가게 합니다."""
        self._nav_target = "clips"

    def _open_clip(self, path):
        if self.clip_path == path and self.clip_cap is not None:
            return
        self._close_clip()
        if path.suffix.lower() in {item.lower() for item in self.settings.photo_extensions}:
            frame = cv2.imread(str(path))
            if frame is None:
                self.register_status.setText("사진을 열 수 없습니다.")
                self._mark_selected()
                return
            self.clip_path = path
            self.clip_cap = None
            self.clip_done = True
            self._set_image(self.clip_image, frame)
            self._mark_selected()
            return
        cap = open_video(path)
        if not cap.isOpened():
            self.register_status.setText("영상을 열 수 없습니다.")
            self._mark_selected()
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.clip_fps = fps if 1 <= fps <= 60 else 15.0
        self.clip_next = 0.0
        self.clip_done = False
        self.clip_cap = cap
        self.clip_path = path
        self._mark_selected()

    def _close_clip(self):
        if self.clip_cap is not None:
            self.clip_cap.release()
        self.clip_cap = None
        self.clip_path = None
        self.clip_done = False
        if hasattr(self, "clip_image"):
            self.clip_image.clear()

    def _play_clip(self):
        if self.clip_cap is None or getattr(self, "clip_done", False):
            return
        now = time.perf_counter()
        if now < self.clip_next:
            return
        self.clip_next = now + 1.0 / self.clip_fps
        ok, frame = self.clip_cap.read()
        if not ok:
            # 한 번만 재생한다. R 키로 처음부터 다시 본다.
            self.clip_done = True
            return
        self._set_image(self.clip_image, frame)

    def _replay_clip(self):
        """저장 영상을 처음부터 다시 재생합니다."""
        if self.recorder.recording or self.clip_path is None:
            return
        path = self.clip_path
        if path.suffix.lower() in {item.lower() for item in self.settings.photo_extensions}:
            frame = cv2.imread(str(path))
            if frame is not None:
                self._set_image(self.clip_image, frame)
            return
        if self.clip_cap is None:
            return
        self.clip_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.clip_next = 0.0
        self.clip_done = False
        ok, frame = self.clip_cap.read()
        if ok:
            self._set_image(self.clip_image, frame)

    def _delete_clip(self):
        """고른 영상·사진 파일을 바로 지웁니다. 확인 창은 띄우지 않습니다."""
        if self.recorder.recording or self.clip_path is None:
            return
        path = self.clip_path
        row = self.clip_list.currentRow()
        # 지우면 바로 위 항목에 포커스. 맨 위면 다음 항목으로 둔다.
        focus = None
        if row > 0:
            focus = Path(self.clip_list.item(row - 1).data(Qt.UserRole))
        elif self.clip_list.count() > 1:
            focus = Path(self.clip_list.item(1).data(Qt.UserRole))
        name = path.name
        self._close_clip()
        if path.exists():
            path.unlink()
        self._nav_target = "clips"
        self._reload_clips(select=focus)
        if self.tag:
            self._refresh_tag(self.tag)
        kind = "사진" if path.suffix.lower() in {item.lower() for item in self.settings.photo_extensions} else "영상"
        self.register_status.setText(f"{kind}을 삭제했습니다. {name}")


def _frame_count(path):
    """저장 영상의 프레임 수. 열리지 않으면 None입니다."""
    cap = open_video(path)
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count if count > 0 else None
