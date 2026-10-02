"""저장한 ROI에서 특징을 뽑아 랜덤포레스트를 학습합니다."""

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gesture.config import settings as default_settings
from gesture.hand import HandAnalyzer


class GestureTrainer:
    """촬영자 단위 GroupKFold 후 모델 전체를 다시 학습해 저장합니다."""

    def __init__(self, settings=None, analyzer=None):
        self.settings = settings or default_settings
        self.analyzer = analyzer or HandAnalyzer(self.settings)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=0)),
            ]
        )

    def run(self):
        samples = self._load()
        if samples is None:
            raise SystemExit("data/ 에 이미지가 없습니다. 먼저 python main.py collect 로 촬영하세요.")
        x, y, groups = samples
        self._report(y)
        self._cross_validate(x, y, groups)
        self.pipeline.fit(x, y)
        self._save()

    def _load(self):
        xs, ys, groups = [], [], []
        for label, gesture in enumerate(self.settings.gestures):
            folder = self.settings.data_dir / gesture
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.png")):
                person = path.stem.split("_", 1)[0]
                image = self._read_bgr(path)
                if image is None:
                    continue
                feat, _mask = self.analyzer.extract(image)
                if feat is None:
                    print("손 분할 실패, 건너뜀:", path)
                    continue
                xs.append(feat)
                ys.append(label)
                groups.append(person)
        if not xs:
            return None
        return np.vstack(xs), np.array(ys), np.array(groups)

    def _report(self, y):
        print("샘플", len(y), "특징", self.analyzer.FEATURE_NAMES)
        for label, gesture in enumerate(self.settings.gestures):
            print(f"  {gesture}: {(y == label).sum()}")

    def _cross_validate(self, x, y, groups):
        people = np.unique(groups)
        if len(people) < 2:
            print("촬영자가 1명이라 GroupKFold는 생략합니다. 다른 사람 데이터를 더 모으세요.")
            return
        splits = min(5, len(people))
        scores = cross_val_score(
            self.pipeline, x, y, cv=GroupKFold(n_splits=splits), groups=groups
        )
        print(f"GroupKFold ({splits}분할, 촬영자 단위) 정확도: {scores.mean():.3f} ± {scores.std():.3f}")

    def _save(self):
        self.settings.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "gestures": list(self.settings.gestures)},
            self.settings.model_path,
        )
        print("저장:", self.settings.model_path)

    def _read_bgr(self, path):
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
