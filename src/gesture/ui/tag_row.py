"""등록 탭의 태그 한 줄."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout


class TagRow(QFrame):
    """태그 이름, 설명, 영상 개수를 한 줄로 보여 줍니다."""

    clicked = Signal(str)

    def __init__(self, tag, count):
        super().__init__()
        self.tag_id = tag["id"]
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(_idle_style())
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
        if tag.get("train"):
            train_label = QLabel("학습")
            train_label.setStyleSheet("font-size: 11px; color: #3b82f6;")
            text.addWidget(train_label)
        self.count_label = QLabel(f"{count}")
        self.count_label.setStyleSheet("font-size: 12px; color: #6e6e73;")
        self.count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addLayout(text, 1)
        layout.addWidget(self.count_label)
        self.set_selected(False)

    def set_count(self, count):
        self.count_label.setText(str(count))

    def set_selected(self, selected):
        self.setStyleSheet(_selected_style() if selected else _idle_style())

    def mousePressEvent(self, event):
        self.clicked.emit(self.tag_id)
        super().mousePressEvent(event)


def _idle_style():
    return """
        QFrame {
            background: #f6f6f7;
            border: 1px solid #e3e3e6;
            border-radius: 8px;
        }
        QLabel { background: transparent; border: none; }
    """


def _selected_style():
    return """
        QFrame {
            background: #e8f2ff;
            border: 1px solid #3b82f6;
            border-radius: 8px;
        }
        QLabel { background: transparent; border: none; }
    """
