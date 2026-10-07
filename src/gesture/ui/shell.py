"""창 닫기와 등록 탭 단축키."""

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QApplication, QWidget

if TYPE_CHECKING:
    from gesture.ui.window import GestureWindow


class Shell(QWidget):
    """등록 탭에서 Space는 녹화, 방향키는 태그 이동입니다."""

    def __init__(self, owner: "GestureWindow"):
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
