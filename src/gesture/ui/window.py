"""인식 탭과 등록 탭을 한 창에 배치합니다."""

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QFrame,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gesture.config import settings as default_settings
from gesture.hand import HandAnalyzer
from gesture.hand_track import HandOverlay
from gesture.model import ModelStore
from gesture.recorder import VideoRecorder
from gesture.smoother import GestureSmoother
from gesture.tags import TagStore
from gesture.ui.camera import CameraPane
from gesture.ui.clips import ClipPane
from gesture.ui.models import ModelPane
from gesture.ui.shell import Shell
from gesture.ui.tags import TagPane



class GestureWindow(TagPane, ModelPane, ClipPane, CameraPane):
    """인식 탭과 등록 탭을 PySide6 창으로 보여 줍니다."""

    def __init__(self, camera=0, settings=None):
        self.camera_index = camera
        self.settings = settings or default_settings
        self.analyzer = HandAnalyzer(self.settings)
        self.hand_overlay = HandOverlay()
        self.smoother = GestureSmoother(self.settings)
        self.recorder = VideoRecorder(self.settings)
        self.tags_store = TagStore(self.settings)
        self.model_store: ModelStore = ModelStore(self.settings, self.analyzer)
        self.tags = []
        self.tag_rows = {}
        self.model = None
        self.model_tags = []
        self.model_features = None
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
        window = Shell(self)
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
        self.hand_overlay.close()
        self.analyzer.close()
        self._close_clip()
