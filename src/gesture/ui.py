"""인식과 등록을 한 창의 두 탭으로 보여 줍니다. UI는 PySide6입니다."""

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gesture.config import settings as default_settings
from gesture.hand import HandAnalyzer
from gesture.model_store import LabelSettings, ModelStore
from gesture.recorder import VideoRecorder
from gesture.smoother import GestureSmoother
from gesture.tags import TagStore


class GestureWindow:
    """인식 탭과 등록 탭을 PySide6 QTabWidget으로 보여 줍니다."""

    def __init__(self, camera=0, settings=None):
        self.camera_index = camera
        self.settings = settings or default_settings
        self.analyzer = HandAnalyzer(self.settings)
        self.smoother = GestureSmoother(self.settings)
        self.recorder = VideoRecorder(self.settings)
        self.tags_store = TagStore(self.settings)
        self.model_store = ModelStore(self.settings, self.analyzer)
        self.tags = []
        self.tag_rows = {}
        self.model = None
        self.model_tags = []
        self._apply_record(self.model_store.active())
        self.cap = None
        self.clip_cap = None
        self.clip_path = None
        self.clip_fps = 15.0
        self.clip_next = 0.0
        self.app = None
        self.window = None
        self.timer = None
        self._closing = False
        self.tag = None

    def run(self):
        self.app = QApplication.instance() or QApplication([])
        self.app.setStyle("Fusion")
        self._build()
        self._open_camera()
        self.timer = QTimer(self.window)
        self.timer.timeout.connect(self._tick)
        self.timer.start(15)
        self.app.installEventFilter(self.window)
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        self.app.exec()

    def _build(self):
        window = _Shell(self)
        window.setWindowTitle("손 제스처")
        window.resize(1180, 760)
        window.setMinimumSize(980, 640)
        self.window = window

        tabs = QTabWidget()
        self.tabs = tabs
        recognize = QWidget()
        register = QWidget()
        tabs.addTab(recognize, "인식")
        tabs.addTab(register, "등록")
        root = QVBoxLayout(window)
        root.setContentsMargins(12, 12, 12, 12)
        root.addWidget(tabs)

        recognize_layout = QVBoxLayout(recognize)
        self.recognize_image = self._video_label()
        self.recognize_status = QLabel("제스처: -")
        recognize_layout.addWidget(self.recognize_image, 1)
        recognize_layout.addWidget(self.recognize_status)

        register_layout = QHBoxLayout(register)

        model_panel = QWidget()
        model_panel.setFixedWidth(180)
        model_layout = QVBoxLayout(model_panel)
        model_layout.setContentsMargins(0, 0, 8, 0)
        model_header = QLabel("모델")
        model_header.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.train_button = QPushButton("학습")
        self.train_button.clicked.connect(self._train)
        self.model_list = QListWidget()
        self.model_list.currentItemChanged.connect(self._select_model)
        model_layout.addWidget(model_header)
        model_layout.addWidget(self.train_button)
        model_layout.addWidget(self.model_list, 1)

        center = QVBoxLayout()
        center.addWidget(QLabel("카메라"))
        self.register_image = self._video_label(180)
        controls = QHBoxLayout()
        self.record_button = QPushButton("녹화")
        self.stop_button = QPushButton("정지(저장)")
        self.stop_button.setEnabled(False)
        self.record_button.clicked.connect(self._start_record)
        self.stop_button.clicked.connect(self._stop_record)
        controls.addWidget(self.record_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        self.register_status = QLabel(
            "태그를 고른 뒤 녹화합니다. 저장 영상은 중앙 박스, 가로 640, 15fps 입니다."
        )
        self.register_status.setWordWrap(True)
        self.clip_image = self._video_label(180)
        center.addWidget(self.register_image, 1)
        center.addLayout(controls)
        center.addWidget(self.register_status)
        center.addWidget(QLabel("저장 영상"))
        center.addWidget(self.clip_image, 1)

        side = QWidget()
        side.setFixedWidth(260)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(8, 0, 0, 0)
        side_layout.setSpacing(8)
        header = QLabel("태그")
        header.setStyleSheet("font-size: 15px; font-weight: 600;")
        side_layout.addWidget(header)

        self.tag_scroll = QScrollArea()
        self.tag_scroll.setWidgetResizable(True)
        self.tag_scroll.setFrameShape(QFrame.NoFrame)
        self.tag_host = QWidget()
        self.tag_layout = QVBoxLayout(self.tag_host)
        self.tag_layout.setContentsMargins(0, 0, 0, 0)
        self.tag_layout.setSpacing(6)
        self.tag_scroll.setWidget(self.tag_host)
        side_layout.addWidget(self.tag_scroll, 1)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.add_button = QPushButton("추가")
        self.edit_button = QPushButton("수정")
        self.delete_button = QPushButton("삭제")
        self.add_button.clicked.connect(self._add_tag)
        self.edit_button.clicked.connect(self._edit_tag)
        self.delete_button.clicked.connect(self._delete_tag)
        actions.addWidget(self.add_button)
        actions.addWidget(self.edit_button)
        actions.addWidget(self.delete_button)
        side_layout.addLayout(actions)

        self.clip_header = QLabel("영상")
        self.clip_header.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.clip_list = QListWidget()
        self.clip_list.currentItemChanged.connect(self._select_clip)
        self.delete_clip_button = QPushButton("영상 삭제")
        self.delete_clip_button.clicked.connect(self._delete_clip)
        side_layout.addWidget(self.clip_header)
        side_layout.addWidget(self.clip_list, 1)
        side_layout.addWidget(self.delete_clip_button)
        self._reload_tags()
        if self.tags and self.tag is None:
            self._choose_tag(self.tags[0]["id"])

        register_layout.addWidget(model_panel)
        register_layout.addLayout(center, 1)
        register_layout.addWidget(side)
        tabs.setCurrentIndex(1)
        self.register_status.setText(
            "Space로 녹화와 저장을 전환합니다. 방향키로 태그를 이동합니다."
        )
        self._reload_models()

    def _video_label(self, min_height=420):
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(min_height)
        label.setStyleSheet("background: black;")
        return label

    def _reload_tags(self, keep=None):
        self.tags = self.tags_store.load()
        if keep is None:
            keep = self.tag
        if keep not in {item["id"] for item in self.tags}:
            keep = None
        self.tag = keep
        while self.tag_layout.count():
            item = self.tag_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.tag_rows = {}
        if not self.tags:
            empty = QLabel("태그가 없습니다. 추가로 만드세요.")
            empty.setWordWrap(True)
            self.tag_layout.addWidget(empty)
        for tag in self.tags:
            row = _TagRow(tag, self._video_count(tag["id"]))
            row.clicked.connect(self._choose_tag)
            self.tag_layout.addWidget(row)
            self.tag_rows[tag["id"]] = row
        self.tag_layout.addStretch(1)
        self._mark_selected()
        self._reload_clips()

    def _choose_tag(self, tag_id):
        if self.recorder.recording:
            return
        self.tag = tag_id
        self._mark_selected()
        self._reload_clips()
        row = self.tag_rows.get(tag_id)
        if row is not None:
            self.tag_scroll.ensureWidgetVisible(row)

    def _move_tag(self, delta):
        if self.recorder.recording or not self.tags:
            return
        ids = [item["id"] for item in self.tags]
        if self.tag not in ids:
            index = 0 if delta > 0 else len(ids) - 1
        else:
            index = (ids.index(self.tag) + delta) % len(ids)
        self._choose_tag(ids[index])

    def _toggle_record(self):
        if self.recorder.recording:
            self._stop_record()
        else:
            self._start_record()

    def _mark_selected(self):
        for tag_id, row in self.tag_rows.items():
            row.set_selected(tag_id == self.tag)
        busy = self.recorder.recording
        chosen = self.tag is not None and not busy
        self.edit_button.setEnabled(chosen)
        self.delete_button.setEnabled(chosen)
        self.add_button.setEnabled(not busy)
        self.train_button.setEnabled(not busy)
        if hasattr(self, "delete_clip_button"):
            self.delete_clip_button.setEnabled(self.clip_path is not None and not busy)

    def _ask_tag(self, title, name="", detail=""):
        dialog = QDialog(self.window)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(360)
        form = QFormLayout(dialog)
        name_edit = QLineEdit(name)
        detail_edit = QLineEdit(detail)
        form.addRow("이름", name_edit)
        form.addRow("설명", detail_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("저장")
        buttons.button(QDialogButtonBox.Cancel).setText("취소")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        name = name_edit.text().strip()
        if not name:
            self.register_status.setText("이름을 입력하세요.")
            return None
        return name, detail_edit.text().strip()

    def _add_tag(self):
        if self.recorder.recording:
            return
        result = self._ask_tag("태그 추가")
        if result is None:
            return
        name, detail = result
        tag = self.tags_store.add(name, detail)
        self._reload_tags(keep=tag["id"])
        self.register_status.setText(f"태그를 추가했습니다. {name}")

    def _edit_tag(self):
        if self.recorder.recording or not self.tag:
            return
        current = next(item for item in self.tags if item["id"] == self.tag)
        result = self._ask_tag("태그 수정", current["title"], current["detail"])
        if result is None:
            return
        name, detail = result
        self.tags_store.update(self.tag, name, detail)
        self._reload_tags(keep=self.tag)
        self.register_status.setText(f"태그를 수정했습니다. {name}")

    def _delete_tag(self):
        if self.recorder.recording or not self.tag:
            return
        current = next(item for item in self.tags if item["id"] == self.tag)
        answer = QMessageBox.question(
            self.window,
            "태그 삭제",
            f"'{current['title']}' 태그를 목록에서 지울까요?\n이미 저장된 영상 파일은 남겨 둡니다.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.tags_store.delete(self.tag)
        self.tag = None
        self._reload_tags()
        self.register_status.setText("태그를 삭제했습니다. 영상 파일은 그대로입니다.")

    def _train(self):
        if self.recorder.recording:
            self.register_status.setText("녹화를 저장한 뒤 학습하세요.")
            return
        self.train_button.setEnabled(False)
        self.register_status.setText("학습 중...")
        QApplication.processEvents()
        try:
            record = self.model_store.train()
        except RuntimeError as exc:
            self.register_status.setText(str(exc))
            self.train_button.setEnabled(True)
            return
        self._apply_record(record)
        self._reload_models()
        self._reload_clips(select=self.clip_path)
        self.register_status.setText(
            f"{record['id']} 저장. 학습 영상 {record['video_count']}개"
        )
        self.train_button.setEnabled(True)

    def _reload_models(self):
        self.model_list.blockSignals(True)
        self.model_list.clear()
        active = self.model_store.active()
        chosen = None
        for record in self.model_store.list_models():
            item = QListWidgetItem(f"{record['id']}    영상 {record['video_count']}개")
            item.setData(Qt.UserRole, record["id"])
            item.setToolTip(record["created"])
            self.model_list.addItem(item)
            if active and record["id"] == active["id"]:
                chosen = item
        if chosen is not None:
            self.model_list.setCurrentItem(chosen)
        self.model_list.blockSignals(False)

    def _select_model(self, current, _previous):
        if current is None:
            return
        model_id = current.data(Qt.UserRole)
        self.model_store.set_active(model_id)
        record = next(item for item in self.model_store.list_models() if item["id"] == model_id)
        self._apply_record(record)

    def _apply_record(self, record):
        if record is None:
            self.model = None
            self.model_tags = []
            self.smoother = GestureSmoother(self.settings)
            return
        pipeline, tags = self.model_store.load(record["id"])
        self.model = pipeline
        self.model_tags = tags
        self.smoother = GestureSmoother(LabelSettings(tags, self.settings))

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

    def _open_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
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
                paths = sorted(folder.glob("*.avi"))
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
        cap = cv2.VideoCapture(str(path))
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

    def _set_image(self, label, frame):
        shown = frame
        height, width = frame.shape[:2]
        if width > 960:
            scale = 960 / width
            shown = cv2.resize(
                frame,
                (960, int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = np.ascontiguousarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
        image_h, image_w, channels = rgb.shape
        image = QImage(rgb.data, image_w, image_h, channels * image_w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy())
        area = label.size()
        if area.width() > 20 and area.height() > 20:
            pixmap = pixmap.scaled(area, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def _title(self, tag):
        for item in self.tags:
            if item["id"] == tag:
                return item["title"]
        return tag

    def _refresh_tag(self, tag):
        row = self.tag_rows.get(tag)
        if row is not None:
            row.set_count(self._video_count(tag))

    def _video_count(self, tag):
        folder = self.settings.video_dir / tag
        if not folder.exists():
            return 0
        return len(list(folder.glob(f"{tag}_*.avi")))

    def _on_close(self):
        if self._closing:
            return
        self._closing = True
        if self.timer is not None:
            self.timer.stop()
        if self.recorder.recording:
            self.recorder.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._close_clip()


class _TagRow(QFrame):
    clicked = Signal(str)

    def __init__(self, tag, count):
        super().__init__()
        self.tag_id = tag["id"]
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            """
            QFrame {
                background: #f6f6f7;
                border: 1px solid #e3e3e6;
                border-radius: 8px;
            }
            QLabel { background: transparent; border: none; }
            """
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title_label = QLabel(tag["title"])
        self.title_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #1c1c1e;")
        self.detail_label = QLabel(tag["detail"] or "설명 없음")
        self.detail_label.setStyleSheet("font-size: 11px; color: #6e6e73;")
        self.detail_label.setWordWrap(True)
        text.addWidget(self.title_label)
        text.addWidget(self.detail_label)
        self.count_label = QLabel(f"{count}")
        self.count_label.setStyleSheet("font-size: 12px; color: #6e6e73;")
        self.count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addLayout(text, 1)
        layout.addWidget(self.count_label)
        self.set_selected(False)

    def set_count(self, count):
        self.count_label.setText(str(count))

    def set_selected(self, selected):
        if selected:
            self.setStyleSheet(
                """
                QFrame {
                    background: #e8f2ff;
                    border: 1px solid #3b82f6;
                    border-radius: 8px;
                }
                QLabel { background: transparent; border: none; }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame {
                    background: #f6f6f7;
                    border: 1px solid #e3e3e6;
                    border-radius: 8px;
                }
                QLabel { background: transparent; border: none; }
                """
            )

    def mousePressEvent(self, event):
        self.clicked.emit(self.tag_id)
        super().mousePressEvent(event)


class _Shell(QWidget):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.installEventFilter(self)

    def closeEvent(self, event):
        self.owner._on_close()
        event.accept()

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.KeyPress or QApplication.activeModalWidget() is not None:
            return super().eventFilter(obj, event)
        if not isinstance(obj, QWidget) or not self.isAncestorOf(obj) and obj is not self:
            return super().eventFilter(obj, event)
        if self.owner.tabs.currentIndex() != 1:
            return super().eventFilter(obj, event)
        key = event.key()
        if key == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self.owner._toggle_record()
            return True
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Left):
            self.owner._move_tag(-1)
            return True
        if key in (Qt.Key.Key_Down, Qt.Key.Key_Right):
            self.owner._move_tag(1)
            return True
        return super().eventFilter(obj, event)
