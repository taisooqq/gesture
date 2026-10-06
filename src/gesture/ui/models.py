"""등록 탭의 모델 목록과 학습 버튼."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidgetItem

from gesture.model.labels import LabelSettings
from gesture.smoother import GestureSmoother


class ModelPane:
    """학습 버튼을 누르고, 고른 모델을 인식에 적용합니다."""

    def _train(self):
        # 녹화 중이면 파일이 아직 닫히지 않았으므로 학습을 시작하지 않는다.
        if self.recorder.recording:
            self.register_status.setText("녹화를 저장한 뒤 학습하세요.")
            return
        # 학습이 끝나는 동안 버튼을 다시 누르지 못하게 끈다.
        self.train_button.setEnabled(False)
        # 화면 아래에 학습이 시작됐다는 문구를 보여 준다.
        self.register_status.setText("학습 중...")
        # 위 문구가 학습 계산에 막히기 전에 화면에 그려지게 한다.
        QApplication.processEvents()
        try:
            # 학습 태그의 영상을 읽어 새 모델 파일을 만들고 목록 기록을 받는다.
            record = self.model_store.train()
        except RuntimeError as exc:
            # 영상이 없거나 태그가 하나뿐이면 그 이유를 화면에 적고 버튼을 되돌린다.
            self.register_status.setText(str(exc))
            self.train_button.setEnabled(True)
            return
        # 방금 만든 모델을 인식 탭이 바로 쓰도록 불러온다.
        self._apply_record(record)
        # 왼쪽 모델 목록을 다시 그려 새 모델을 선택 상태로 만든다.
        self._reload_models()
        # 영상 목록의 학습됨 표시를 방금 학습에 쓴 파일 기준으로 고친다.
        self._reload_clips(select=self.clip_path)
        # 저장된 모델 이름과 그 모델이 학습한 영상 개수를 보여 준다.
        self.register_status.setText(
            f"{record['id']} 저장. 학습 영상 {record['video_count']}개"
        )
        # 다음 학습을 누를 수 있게 버튼을 다시 켠다.
        self.train_button.setEnabled(True)

    def _reload_models(self):
        # 목록을 비우는 동안 선택 변경 신호가 인식을 바꾸지 않게 막는다.
        self.model_list.blockSignals(True)
        # 화면에 있던 모델 줄을 모두 지운다.
        self.model_list.clear()
        # 지금 인식에 쓰는 모델 기록을 가져온다.
        active = self.model_store.active()
        # 목록을 다시 채운 뒤 선택할 줄을 담아 둔다.
        chosen = None
        # 저장된 모델을 만든 순서대로 한 줄씩 넣는다.
        for record in self.model_store.list_models():
            # 보이는 글자는 모델 이름과 학습 영상 개수다.
            item = QListWidgetItem(f"{record['id']}    영상 {record['video_count']}개")
            # 줄을 눌렀을 때 찾을 모델 이름을 줄 안에 숨겨 둔다.
            item.setData(Qt.UserRole, record["id"])
            # 마우스를 올리면 그 모델을 만든 시각이 나오게 한다.
            item.setToolTip(record["created"])
            # 왼쪽 목록에 이 줄을 붙인다.
            self.model_list.addItem(item)
            # 인식 중인 모델과 같은 줄이면 나중에 선택한다.
            if active and record["id"] == active["id"]:
                chosen = item
        # 활성 모델 줄이 있으면 그 줄을 선택된 상태로 만든다.
        if chosen is not None:
            self.model_list.setCurrentItem(chosen)
        # 막았던 선택 신호를 다시 받게 한다.
        self.model_list.blockSignals(False)

    def _select_model(self, current, _previous):
        # 선택한 줄이 없으면 바꿀 모델이 없다.
        if current is None:
            return
        # 줄에 숨겨 둔 모델 이름을 꺼낸다.
        model_id = current.data(Qt.UserRole)
        # 다음에 앱을 열어도 이 모델을 쓰도록 목록 파일에 적는다.
        self.model_store.set_active(model_id)
        # 그 이름의 전체 기록을 목록에서 찾는다.
        record = next(item for item in self.model_store.list_models() if item["id"] == model_id)
        # 찾은 모델을 인식에 적용한다.
        self._apply_record(record)

    def _apply_record(self, record):
        # 저장된 모델이 없으면 인식은 손 없음만 내도록 비운다.
        if record is None:
            self.model = None
            self.model_tags = []
            self.smoother = GestureSmoother(self.settings)
            return
        # 모델 파일에서 분류기와 그 모델이 배운 태그 순서를 읽는다.
        pipeline, tags = self.model_store.load(record["id"])
        # 웹캠 판정이 이 분류기를 쓰게 한다.
        self.model = pipeline
        # 분류기 출력 순서를 태그 이름과 맞추기 위해 저장한다.
        self.model_tags = tags
        # 흔들린 판정을 거르는 기준을 이 태그의 개수에 맞춘다.
        self.smoother = GestureSmoother(LabelSettings(tags, self.settings))
