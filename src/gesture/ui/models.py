"""등록 탭의 모델 목록과 학습 버튼."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidgetItem

from gesture.model.labels import LabelSettings
from gesture.smoother import GestureSmoother


class ModelPane:
    """학습 버튼을 누르고, 고른 모델을 인식에 적용합니다."""

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
