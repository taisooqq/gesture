"""저장한 영상으로 모델을 만들고, 모델마다 영상 개수를 기록합니다."""

import json
import re
from datetime import datetime

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gesture.config import settings as default_settings
from gesture.hand import HandAnalyzer
from gesture.recorder import open_video
from gesture.tags import TagStore


class ModelStore:
    """모델 파일은 models/model_01.joblib, 목록은 models/catalog.json 입니다."""

    def __init__(self, settings=None, analyzer=None):
        self.settings = settings or default_settings
        self.analyzer = analyzer or HandAnalyzer(self.settings)
        self.directory = self.settings.model_path.parent
        self.catalog_path = self.directory / "catalog.json"

    def list_models(self):
        catalog = self._read_catalog()
        items = []
        for item in catalog["models"]:
            if (self.directory / item["file"]).exists():
                items.append(item)
        return items

    def active(self):
        catalog = self._read_catalog()
        active_id = catalog.get("active")
        for item in self.list_models():
            if item["id"] == active_id:
                return item
        models = self.list_models()
        return models[-1] if models else None

    def set_active(self, model_id):
        catalog = self._read_catalog()
        catalog["active"] = model_id
        self._write_catalog(catalog)

    def trained_videos(self):
        found = set()
        for item in self.list_models():
            for path in item.get("videos", []):
                found.add(path)
        return found

    def load(self, model_id):
        catalog = self._read_catalog()
        item = next(row for row in catalog["models"] if row["id"] == model_id)
        blob = joblib.load(self.directory / item["file"])
        return blob["pipeline"], list(blob["tags"]), blob.get("features")

    def train(self):
        samples = self._samples()
        if samples is None:
            raise RuntimeError("저장된 영상이 없습니다. 먼저 태그를 골라 녹화하세요.")
        features, labels, tags, videos, tag_counts = samples
        if len(tags) < 2:
            raise RuntimeError("태그가 다른 영상이 2종류 이상 있어야 학습할 수 있습니다.")
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=0)),
            ]
        )
        # 위에서 만든 관절 숫자와 태그 이름을 랜덤 포레스트가 나눈다.
        pipeline.fit(features, labels)
        return self._save(pipeline, tags, videos, tag_counts)

    def _samples(self):
        # 태그 목록에서 train 이 켜진 태그의 id 만 모은다. 폴더 이름과 같은 값이다.
        allowed = set(TagStore(self.settings).train_ids())
        # 학습에 넣을 (태그 이름, avi 목록). 이름 순으로 쌓이므로 이 순서가 곧 클래스 번호가 된다.
        folders = []
        # 영상 폴더는 있는데 학습 태그와 안 맞는지, 아예 영상이 없는지 나중에 구분하려고 둔다.
        saw_video = False
        # data/videos 가 없으면 아래 루프를 건너뛰고, 영상이 없다는 쪽으로 빠진다.
        if self.settings.video_dir.exists():
            # 태그 폴더를 이름 순으로 돌린다. 순서가 바뀌면 같은 영상도 라벨 번호가 달라진다.
            for folder in sorted(self.settings.video_dir.iterdir()):
                # 태그 폴더가 아닌 파일은 건너뛴다.
                if not folder.is_dir():
                    continue
                # 그 태그 폴더 안의 avi 만, 파일 이름 순으로 고른다.
                videos = sorted(folder.glob("*.avi"))
                # 빈 폴더는 영상이 있는 것으로 세지 않는다.
                if not videos:
                    continue
                # avi 가 하나라도 있으면, 학습 태그 여부와 관계없이 "영상은 있다"고 표시한다.
                saw_video = True
                # 폴더 이름이 학습 태그 id 일 때만 학습 대상으로 넣는다.
                if folder.name in allowed:
                    folders.append((folder.name, videos))
        # 학습 태그에 해당하는 폴더가 하나도 없을 때.
        if not folders:
            # 다른 태그 폴더에 영상은 있는데, 학습으로 켠 태그에는 없다.
            if saw_video:
                raise RuntimeError("학습으로 지정된 태그에 영상이 없습니다.")
            # 녹화 영상 자체가 없다. train() 이 이 None 을 보고 녹화 안내를 낸다.
            return None
        # 클래스 이름 목록. 폴더를 넣은 순서 그대로다.
        tags = [name for name, _videos in folders]
        # 한 행이 1초 묶음의 특징 벡터. 영상 한 장당 한 행이 아니다.
        xs = []
        # xs 와 같은 순서의 클래스 번호. 0 이 tags[0], 1 이 tags[1] 이다.
        ys = []
        # 태그별로 손을 찾은 영상 개수. 아직 0 이고, 특징이 나온 영상만 올린다.
        tag_counts = {name: 0 for name in tags}
        # 태그별로 실제로 학습에 쓴 영상 경로. 카탈로그 videos 필드에 남긴다.
        used_by_tag = {name: [] for name in tags}
        # enumerate 의 번호가 그 태그의 클래스 번호다.
        for label, (name, videos) in enumerate(folders):
            for path in videos:
                # 이 영상에서 손을 찾은 1초 묶음이 하나라도 있으면 True.
                used = False
                # 학습 입력은 hand.py 의 window_features 다.
                # 1초 묶음마다 손목 기준 다섯 손가락 끝의 위치와 이동을 한 줄로 넣는다.
                # 15장 묶음을 5장씩 밀며 자른다. 1초보다 짧은 영상은 통째로 한 묶음이다.
                for window in _sample_windows(path):
                    # 묶음 안 프레임의 절반 이상에서 손을 못 찾으면 None.
                    feat = self.analyzer.window_features(window)
                    if feat is None:
                        continue
                    # 관절 21개의 (x, y, dx, dy) 를 한 줄로 편 벡터. 길이는 84.
                    xs.append(feat)
                    # 이 벡터의 정답은 지금 폴더의 클래스 번호다.
                    ys.append(label)
                    used = True
                # 묶음이 하나도 안 남은 영상은 개수와 경로 목록에서 뺀다.
                if used:
                    tag_counts[name] += 1
                    # data/videos 기준 상대 경로. 예: left_click/left_click_00e228.avi
                    used_by_tag[name].append(path.relative_to(self.settings.video_dir).as_posix())
        # 손을 한 번도 못 찾은 태그는 개수 기록에서 지운다.
        tag_counts = {name: count for name, count in tag_counts.items() if count}
        # 모든 영상, 모든 묶음에서 특징이 하나도 안 나왔을 때.
        if not xs:
            # 손 좌표 모델 파일을 열지 못한 경우 그 메시지를 그대로 올린다.
            if self.analyzer.hands_error:
                raise RuntimeError(self.analyzer.hands_error)
            # 모델은 열렸지만 화면에서 손을 못 찾았다.
            raise RuntimeError("영상에서 손을 찾지 못했습니다. 중앙 박스 안에서 다시 녹화하세요.")
        # 특징이 남은 태그만 클래스 이름으로 남긴다. 순서는 원래 tags 순서를 유지한다.
        tags = [name for name in tags if tag_counts.get(name)]
        # 손이 없는 태그가 빠져 클래스 개수가 줄었으면, 남은 번호가 0부터 이어지게 다시 매긴다.
        if len(tags) != len(folders):
            # 살아남은 태그 이름 -> 새 클래스 번호.
            kept = {name: index for index, name in enumerate(tags)}
            # 특징을 만들 때 쓰던 번호가 가리키던 원래 태그 이름.
            old_names = [name for name, _videos in folders]
            remapped = []
            filtered_x = []
            for feat, label in zip(xs, ys):
                name = old_names[label]
                # 그 태그가 빠졌으면 이 행도 학습에서 뺀다.
                if name in kept:
                    filtered_x.append(feat)
                    # 예전 번호 대신 살아남은 태그만으로 다시 센 번호를 붙인다.
                    remapped.append(kept[name])
            xs, ys = filtered_x, remapped
        # 카탈로그에 남길 영상 경로. 클래스 순서대로 이어 붙인다.
        trained_videos = []
        for name in tags:
            trained_videos.extend(used_by_tag[name])
        # xs 는 (샘플 수, 84), ys 는 (샘플 수,) 정수 배열.
        # tags 는 클래스 이름, trained_videos 는 쓴 영상 경로, tag_counts 는 태그별 영상 수.
        return np.vstack(xs), np.array(ys), tags, trained_videos, tag_counts

    def _save(self, pipeline, tags, videos, tag_counts):
        self.directory.mkdir(parents=True, exist_ok=True)
        catalog = self._read_catalog()
        model_id = self._next_id(catalog)
        filename = f"{model_id}.joblib"
        joblib.dump(
            {
                "pipeline": pipeline,
                "tags": tags,
                "video_count": len(videos),
                "videos": videos,
                "features": HandAnalyzer.FEATURE_KIND,
            },
            self.directory / filename,
        )
        record = {
            "id": model_id,
            "file": filename,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "video_count": len(videos),
            "videos": videos,
            "tag_counts": tag_counts,
        }
        catalog["models"].append(record)
        catalog["active"] = model_id
        self._write_catalog(catalog)
        return record

    def _next_id(self, catalog):
        numbers = []
        for item in catalog["models"]:
            match = re.fullmatch(r"model_(\d+)", item["id"])
            if match:
                numbers.append(int(match.group(1)))
        return f"model_{max(numbers, default=0) + 1:02d}"

    def _read_catalog(self):
        if not self.catalog_path.exists():
            return {"active": None, "models": []}
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        data.setdefault("active", None)
        data.setdefault("models", [])
        return data

    def _write_catalog(self, catalog):
        self.directory.mkdir(parents=True, exist_ok=True)
        text = json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"
        self.catalog_path.write_text(text, encoding="utf-8")


# 학습 영상을 연속 1초 묶음으로 자른다. 15fps면 15장이 1초이고, 5장마다 다음 묶음을 만든다.
def _sample_windows(path, length=15, stride=5):
    cap = open_video(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < length:
        if frames:
            yield frames
        return
    for start in range(0, len(frames) - length + 1, stride):
        yield frames[start : start + length]
