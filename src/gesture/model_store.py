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
        return blob["pipeline"], list(blob["tags"])

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
        pipeline.fit(features, labels)
        return self._save(pipeline, tags, videos, tag_counts)

    def _samples(self):
        folders = []
        if self.settings.video_dir.exists():
            for folder in sorted(self.settings.video_dir.iterdir()):
                if folder.is_dir():
                    videos = sorted(folder.glob("*.avi"))
                    if videos:
                        folders.append((folder.name, videos))
        if not folders:
            return None
        tags = [name for name, _videos in folders]
        xs = []
        ys = []
        tag_counts = {name: 0 for name in tags}
        used_by_tag = {name: [] for name in tags}
        for label, (name, videos) in enumerate(folders):
            for path in videos:
                used = False
                for frame in _sample_frames(path):
                    feat, _mask = self.analyzer.extract(frame)
                    if feat is None:
                        continue
                    xs.append(feat)
                    ys.append(label)
                    used = True
                if used:
                    tag_counts[name] += 1
                    used_by_tag[name].append(path.relative_to(self.settings.video_dir).as_posix())
        tag_counts = {name: count for name, count in tag_counts.items() if count}
        if not xs:
            raise RuntimeError("영상에서 손을 찾지 못했습니다. 중앙 박스 안에서 다시 녹화하세요.")
        tags = [name for name in tags if tag_counts.get(name)]
        if len(tags) != len(folders):
            kept = {name: index for index, name in enumerate(tags)}
            old_names = [name for name, _videos in folders]
            remapped = []
            filtered_x = []
            for feat, label in zip(xs, ys):
                name = old_names[label]
                if name in kept:
                    filtered_x.append(feat)
                    remapped.append(kept[name])
            xs, ys = filtered_x, remapped
        trained_videos = []
        for name in tags:
            trained_videos.extend(used_by_tag[name])
        return np.vstack(xs), np.array(ys), tags, trained_videos, tag_counts

    def _save(self, pipeline, tags, videos, tag_counts):
        self.directory.mkdir(parents=True, exist_ok=True)
        catalog = self._read_catalog()
        model_id = self._next_id(catalog)
        filename = f"{model_id}.joblib"
        joblib.dump(
            {"pipeline": pipeline, "tags": tags, "video_count": len(videos), "videos": videos},
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


def _sample_frames(path, limit=20):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        indexes = np.linspace(0, max(total - 1, 0), num=min(limit, total), dtype=int)
        for index in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    else:
        while len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    cap.release()
    return frames


class LabelSettings:
    """영상 태그 수에 맞춰 스무더가 읽을 이름 목록입니다."""

    def __init__(self, tags, settings):
        self.gestures = tuple(tags)
        self.actions = {}
        self.ema_alpha = settings.ema_alpha
        self.hold_frames = settings.hold_frames
        self.conf_threshold = settings.conf_threshold
