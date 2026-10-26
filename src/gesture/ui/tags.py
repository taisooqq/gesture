"""등록 탭의 태그 목록과 추가·수정·삭제."""

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QRadioButton,
    QSpinBox,
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
        self._nav_target = "tags"
        self.tag = tag_id
        self._mark_selected()
        self._reload_clips()
        if hasattr(self, "_update_capture_ui"):
            self._update_capture_ui()
        row = self.tag_rows.get(tag_id)
        if row is not None:
            self.tag_scroll.ensureWidgetVisible(row)

    def _move_nav(self, delta):
        """마지막에 클릭한 쪽(태그 또는 영상)을 방향키로 옮깁니다."""
        if getattr(self, "_nav_target", "tags") == "clips":
            self._move_clip(delta)
            return
        self._move_tag(delta)

    def _move_tag(self, delta):
        if self.recorder.recording or not self.tags:
            return
        self._nav_target = "tags"
        ids = [item["id"] for item in self.tags]
        if self.tag not in ids:
            index = 0 if delta > 0 else len(ids) - 1
        else:
            index = (ids.index(self.tag) + delta) % len(ids)
        self._choose_tag(ids[index])

    def _move_clip(self, delta):
        if self.recorder.recording or not hasattr(self, "clip_list"):
            return
        count = self.clip_list.count()
        if count <= 0:
            return
        self._nav_target = "clips"
        row = self.clip_list.currentRow()
        if row < 0:
            row = 0 if delta > 0 else count - 1
        else:
            row = (row + delta) % count
        self.clip_list.setCurrentRow(row)
        item = self.clip_list.currentItem()
        if item is not None:
            self.clip_list.scrollToItem(item)

    def _mark_selected(self):
        for tag_id, row in self.tag_rows.items():
            row.set_selected(tag_id == self.tag)
        busy = self.recorder.recording
        chosen = self.tag is not None and not busy
        self.edit_button.setEnabled(chosen)
        self.delete_button.setEnabled(chosen)
        self.add_button.setEnabled(not busy)
        self.train_button.setEnabled(not busy)
        if hasattr(self, "score_button"):
            self.score_button.setEnabled(not busy)
        if hasattr(self, "cnn_button"):
            self.cnn_button.setEnabled(not busy)
        if hasattr(self, "cnn_score_button"):
            self.cnn_score_button.setEnabled(not busy)
        if hasattr(self, "delete_clip_button"):
            self.delete_clip_button.setEnabled(self.clip_path is not None and not busy)
        if hasattr(self, "_update_capture_ui"):
            self._update_capture_ui()

    def _ask_tag(
        self,
        title,
        name="",
        detail="",
        train=False,
        threshold=None,
        folder_id="",
        photo=False,
        create=False,
    ):
        dialog = QDialog(self.window)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(400)
        form = QFormLayout(dialog)
        kind_row = QHBoxLayout()
        video_radio = QRadioButton("동작 (영상)")
        photo_radio = QRadioButton("무동작 (사진)")
        kind_group = QButtonGroup(dialog)
        kind_group.addButton(video_radio)
        kind_group.addButton(photo_radio)
        (photo_radio if photo else video_radio).setChecked(True)
        kind_row.addWidget(video_radio)
        kind_row.addWidget(photo_radio)
        kind_row.addStretch(1)
        form.addRow("종류", kind_row)
        folder_edit = QLineEdit(folder_id)
        folder_edit.setPlaceholderText("예: idle_fist  (영문 소문자, 숫자, _)")
        folder_edit.setToolTip("data/videos/ 아래 폴더 이름입니다. 만든 뒤에는 바꿀 수 없습니다.")
        if create:
            form.addRow("폴더명", folder_edit)
        else:
            folder_label = QLabel(folder_id or "(없음)")
            folder_label.setStyleSheet("color: #6e6e73;")
            form.addRow("폴더명", folder_label)
        name_edit = QLineEdit(name)
        name_edit.setPlaceholderText("예: 무동작 주먹")
        detail_edit = QLineEdit(detail)
        train_box = QCheckBox("학습에 사용")
        train_box.setChecked(train)
        score = QSpinBox()
        score.setRange(1, 99)
        score.setSuffix("%")
        if threshold is None:
            threshold = self.settings.conf_threshold
        score.setValue(round(float(threshold) * 100))
        score.setToolTip("이 점수 이상일 때만 이 동작으로 봅니다. 다음 학습부터 적용됩니다.")
        form.addRow("이름", name_edit)
        form.addRow("설명", detail_edit)
        form.addRow("학습", train_box)
        form.addRow("인식 점수", score)
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
        return {
            "title": name,
            "detail": detail_edit.text().strip(),
            "train": train_box.isChecked(),
            "threshold": score.value() / 100,
            "photo": photo_radio.isChecked(),
            "folder_id": folder_edit.text().strip() if create else folder_id,
        }

    def _add_tag(self):
        if self.recorder.recording:
            return
        result = self._ask_tag("태그 추가", create=True)
        if result is None:
            return
        try:
            tag = self.tags_store.add(
                result["title"],
                result["detail"],
                result["train"],
                result["threshold"],
                folder_id=result["folder_id"] or None,
                photo=result["photo"],
            )
        except ValueError as exc:
            self.register_status.setText(str(exc))
            return
        self._reload_tags(keep=tag["id"])
        kind = "사진 무동작" if tag.get("photo") else "영상"
        self.register_status.setText(f"태그를 추가했습니다. {tag['title']} · {kind} · {tag['id']}")

    def _edit_tag(self):
        if self.recorder.recording or not self.tag:
            return
        current = next(item for item in self.tags if item["id"] == self.tag)
        result = self._ask_tag(
            "태그 수정",
            current["title"],
            current["detail"],
            current.get("train", False),
            current.get("threshold", self.settings.conf_threshold),
            folder_id=current["id"],
            photo=bool(current.get("photo")),
            create=False,
        )
        if result is None:
            return
        self.tags_store.update(
            self.tag,
            result["title"],
            result["detail"],
            result["train"],
            result["threshold"],
            photo=result["photo"],
        )
        self._reload_tags(keep=self.tag)
        self.register_status.setText(f"태그를 수정했습니다. {result['title']}")

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
        photo = any(item["id"] == tag and item.get("photo") for item in self.tags)
        if photo:
            count = 0
            for extension in self.settings.photo_extensions:
                count += len(list(folder.glob(f"*{extension}")))
            return count
        return len(list(folder.glob(f"{tag}_*.avi")))
