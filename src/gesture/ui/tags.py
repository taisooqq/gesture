"""등록 탭의 태그 목록과 추가·수정·삭제."""

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
)

from gesture.recorder import VideoRecorder
from gesture.tags import TagStore
from gesture.ui.tag_row import TagRow


class TagPane:
    """선택한 태그와 태그 줄의 영상 개수를 다룹니다."""

    tags_store: TagStore
    recorder: VideoRecorder

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
            row = TagRow(tag, self._video_count(tag["id"]))
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

    def _ask_tag(self, title, name="", detail="", train=False):
        dialog = QDialog(self.window)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(360)
        form = QFormLayout(dialog)
        name_edit = QLineEdit(name)
        detail_edit = QLineEdit(detail)
        train_box = QCheckBox("학습에 사용")
        train_box.setChecked(train)
        form.addRow("이름", name_edit)
        form.addRow("설명", detail_edit)
        form.addRow("학습", train_box)
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
        return name, detail_edit.text().strip(), train_box.isChecked()

    def _add_tag(self):
        if self.recorder.recording:
            return
        result = self._ask_tag("태그 추가")
        if result is None:
            return
        name, detail, train = result
        tag = self.tags_store.add(name, detail, train)
        self._reload_tags(keep=tag["id"])
        self.register_status.setText(f"태그를 추가했습니다. {name}")

    def _edit_tag(self):
        if self.recorder.recording or not self.tag:
            return
        current = next(item for item in self.tags if item["id"] == self.tag)
        result = self._ask_tag("태그 수정", current["title"], current["detail"], current.get("train", False))
        if result is None:
            return
        name, detail, train = result
        self.tags_store.update(self.tag, name, detail, train)
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
