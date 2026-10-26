"""저장한 영상으로 모델을 만들고, 모델마다 영상 개수를 기록합니다."""

import json
import random
import re
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gesture.config import settings as default_settings
from gesture.hand import HandAnalyzer
from gesture.model.landmarks import video_points
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
        if item.get("kind") == "motion":
            from gesture.model.sequence import load_motion_model

            model, tags, thresholds = load_motion_model(self.directory / item["file"])
            return model, tags, "motion", thresholds
        if item.get("kind") == "cnn1d":
            from gesture.model.cnn import load_sequence_model

            model, tags, thresholds = load_sequence_model(self.directory / item["file"])
            return model, tags, "cnn1d", thresholds
        blob = joblib.load(self.directory / item["file"])
        return blob["pipeline"], list(blob["tags"]), blob.get("features"), blob.get("thresholds") or {}

    # 학습
    def train(self):
        samples = self._samples()
        if samples is None:
            raise RuntimeError("저장된 영상이 없습니다. 먼저 태그를 골라 녹화하세요.")
        features, labels, tags, videos, tag_counts = samples
        if len(tags) < 2:
            raise RuntimeError("태그가 다른 영상이 2종류 이상 있어야 학습할 수 있습니다.")
        pipeline = _forest()
        # 위에서 만든 관절 숫자와 태그 이름을 랜덤 포레스트가 나눈다.
        pipeline.fit(features, labels)
        return self._save(pipeline, tags, videos, tag_counts, self._thresholds_for(tags))

    def train_motion(self):
        """같은 학습 영상으로 딥러닝 모델을 만듭니다. 랜덤포레스트 모델은 바꾸지 않습니다."""
        train_motion_model, save_motion_model = _torch_parts()
        folders = self._tagged_videos(include_photos=True)
        if folders is None:
            raise RuntimeError("저장된 영상이 없습니다. 먼저 태그를 골라 녹화하세요.")
        clips, tags, videos, tag_counts = self._motion_clips(folders)
        model = train_motion_model([clip[:5] for clip in clips], tags, self.settings)
        thresholds = self._thresholds_for(tags)
        return self._save_motion(model, videos, tag_counts, thresholds, save_motion_model)

    def evaluate(self):
        """태그마다 영상의 20%를 빼고 맞힌 비율을 계산합니다. 모델은 저장하지 않습니다."""
        train_groups, test_groups, skipped = self._split_folders()
        trained = self._rows(train_groups, augment=True)
        if trained is None:
            raise RuntimeError("학습으로 쓸 영상에서 손을 찾지 못했습니다.")
        features, labels, tags, _videos, _counts = trained
        if len(tags) < 2:
            raise RuntimeError("손이 잡힌 태그가 2종류 이상 있어야 인식률을 볼 수 있습니다.")
        pipeline = _forest()
        pipeline.fit(features, labels)
        thresholds = self._thresholds_for(tags)
        tests = []
        for name, videos in test_groups:
            points = video_points(self.settings, videos)
            per_video = []
            for path in videos:
                frames, _aspect = points[path]
                feats = [self.analyzer.window_features_points(window) for window in _windows(frames)]
                per_video.append([feat for feat in feats if feat is not None])
            tests.append((name, per_video))

        def guess(feat):
            proba = pipeline.predict_proba(feat.reshape(1, -1))[0]
            return self._guess(proba, pipeline.classes_, tags, thresholds)

        return self._score_report(tags, tests, skipped, guess)

    def evaluate_motion(self):
        """딥러닝으로 태그마다 영상의 20%를 빼고 맞힌 비율을 계산합니다. 모델은 저장하지 않습니다."""
        from gesture.model.motion import features

        train_motion_model, _save = _torch_parts()
        train_groups, test_groups, skipped = self._split_folders(include_photos=True)
        clips, tags, _videos, _counts = self._motion_clips(train_groups)
        model = train_motion_model([clip[:5] for clip in clips], tags, self.settings)
        thresholds = self._thresholds_for(tags)

        def guess(sequence):
            return self._guess(model.predict_proba(sequence), model.classes_, tags, thresholds)

        test_folders = [(name, videos) for name, videos in test_groups if name in tags]
        skipped.extend(name for name, _videos in test_groups if name not in tags)
        test_clips, _tags, _used, _counts = self._motion_clips(test_folders, tags=tags, allow_short=True)
        tests = {}
        held = [0, 0]
        rng = np.random.default_rng(1)
        for label, joints, depth, aspect, moving, video in test_clips:
            tests.setdefault(tags[label], {}).setdefault(video, []).append(features(joints, depth, aspect))
            if not moving:
                continue
            # 움직여야 하는 동작의 처음·끝 자세로 멈춘 손은 무동작이어야 한다.
            for index in (0, joints.shape[0] - 1):
                still_j = np.repeat(joints[index : index + 1], self.settings.seq_window, axis=0)
                still_d = np.repeat(depth[index : index + 1], self.settings.seq_window, axis=0)
                scale = float(np.linalg.norm(joints[index, 9] - joints[index, 0]))
                still_j = still_j + rng.normal(0.0, 0.01 * scale, size=still_j.shape).astype(np.float32)
                held[1] += 1
                if guess(features(still_j, still_d, aspect)) in (None, "idle"):
                    held[0] += 1
        grouped = [(name, list(videos.values())) for name, videos in tests.items()]
        extra = None
        if held[1]:
            extra = f"멈춘 손을 무동작으로 본 비율 {round(100 * held[0] / held[1])}%  {held[0]}/{held[1]}"
        return self._score_report([name for name in tags if name != "idle"], grouped, skipped, guess, extra)

    def _split_folders(self, include_photos=False):
        folders = self._tagged_videos(include_photos=include_photos)
        if folders is None:
            raise RuntimeError("저장된 영상이 없습니다. 먼저 태그를 골라 녹화하세요.")
        train_groups = []
        test_groups = []
        skipped = []
        for name, videos in folders:
            if len(videos) < 2:
                skipped.append(name)
                continue
            train_paths, test_paths = _split_videos(videos)
            train_groups.append((name, train_paths))
            test_groups.append((name, test_paths))
        if len(train_groups) < 2:
            raise RuntimeError("인식률을 보려면 영상이 2개 이상인 태그가 2종류 있어야 합니다.")
        return train_groups, test_groups, skipped

    def _score_report(self, tags, tests, skipped, guess, extra=None):
        """시험 영상을 증강 없이 맞춰 보고, 태그별 비율을 만듭니다. tests 는 (태그, 영상별 묶음 목록) 입니다."""
        titles = {tag["id"]: tag["title"] for tag in TagStore(self.settings).load()}
        titles.setdefault("idle", "무동작")
        totals = {name: 0 for name in tags}
        hits = {name: 0 for name in tags}
        confused = {name: {} for name in tags}
        test_videos = 0
        kept = set(tags)
        for name, per_video in tests:
            if name not in kept:
                skipped.append(name)
                continue
            for feats in per_video:
                if not feats:
                    continue
                test_videos += 1
                for feat in feats:
                    totals[name] += 1
                    predicted = guess(feat)
                    if predicted == "idle":
                        predicted = None
                    if predicted == name:
                        hits[name] += 1
                    else:
                        confused[name][predicted] = confused[name].get(predicted, 0) + 1
        for name in tags:
            if totals[name] == 0 and name not in skipped:
                skipped.append(name)
        total = sum(totals.values())
        if total == 0:
            raise RuntimeError("시험 영상에서 손을 찾지 못했습니다.")
        hit = sum(hits.values())
        summary = f"전체 {round(100 * hit / total)}%  시험 묶음 {total}개, 영상 {test_videos}개"
        lines = [summary]
        if extra:
            lines.append(extra)
        lines.append("")
        for name in tags:
            if totals[name] == 0:
                continue
            title = titles.get(name, name)
            lines.append(f"{title}  {round(100 * hits[name] / totals[name])}%  {hits[name]}/{totals[name]}")
            wrong = sorted(confused[name].items(), key=lambda item: item[1], reverse=True)
            if wrong:
                parts = []
                for other, count in wrong:
                    label = "무동작" if other in (None, "idle") else titles.get(other, other)
                    parts.append(f"{label} {count}")
                lines.append("  다른 판정: " + ", ".join(parts))
        if skipped:
            names = ", ".join(titles.get(name, name) for name in dict.fromkeys(skipped))
            lines.extend(["", f"영상이 적거나 손을 못 찾아 빼 둠: {names}"])
        if extra:
            summary = f"{summary} · {extra.split('  ')[0]}"
        return {"summary": summary, "detail": "\n".join(lines)}

    def _guess(self, proba, classes, tags, thresholds):
        """기준점 미만이면 무동작입니다. 무동작은 None 으로 돌려줍니다."""
        index = int(np.argmax(proba))
        name = tags[int(classes[index])]
        if float(proba[index]) < thresholds.get(name, self.settings.conf_threshold):
            return None
        return name

    def _samples(self):
        folders = self._tagged_videos(include_photos=False)
        if folders is None:
            return None
        return self._rows(folders, augment=True)

    def _tagged_videos(self, include_photos=False):
        # 태그 목록에서 train 이 켜진 태그의 id 만 모은다. 폴더 이름과 같은 값이다.
        allowed = set(TagStore(self.settings).train_ids())
        photo_tags = set(self.settings.photo_tags)
        # 학습에 넣을 (태그 이름, 파일 목록). 이름 순으로 쌓이므로 이 순서가 곧 클래스 번호가 된다.
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
                # 무동작 등은 사진, 나머지는 avi. 랜덤포레스트는 사진을 쓰지 않는다.
                if folder.name in photo_tags:
                    if not include_photos:
                        continue
                    videos = []
                    for extension in self.settings.photo_extensions:
                        videos.extend(folder.glob(f"*{extension}"))
                    videos = sorted(videos)
                else:
                    videos = sorted(folder.glob("*.avi"))
                # 빈 폴더는 영상이 있는 것으로 세지 않는다.
                if not videos:
                    continue
                # 파일이 하나라도 있으면, 학습 태그 여부와 관계없이 "자료는 있다"고 표시한다.
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
        return folders

    def _rows(self, folders, augment):
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
        # 영상마다 손 좌표는 한 번만 뽑아 data/landmarks 에 둔다. 처음에는 코어마다 나눠 뽑는다.
        points = video_points(self.settings, [path for _name, videos in folders for path in videos])
        # enumerate 의 번호가 그 태그의 클래스 번호다.
        for label, (name, videos) in enumerate(folders):
            for path in videos:
                # 이 영상에서 손을 찾은 1초 묶음이 하나라도 있으면 True.
                used = False
                frames, _aspect = points[path]
                # 학습 입력은 hand.py 의 window_features 다.
                # 1초 묶음마다 손가락 끝, 엄지·검지·중지 깊이, 굽은 각, 중지 10번 좌표를 한 줄로 넣는다.
                # 15장 묶음을 5장씩 밀며 자른다. 1초보다 짧은 영상은 통째로 한 묶음이다.
                for window in _windows(frames):
                    # 원본, 위가 넓은 사다리꼴 두 벌, 아래가 넓은 사다리꼴 두 벌에 각도·속도·흔들림을 넣는다.
                    # 손이 거의 없으면 빈 목록이다. 모델 7까지는 이 과정이 없다.
                    if augment:
                        feats = self.analyzer.augmented_features_points(window)
                    else:
                        feat = self.analyzer.window_features_points(window)
                        feats = [] if feat is None else [feat]
                    for feat in feats:
                        xs.append(feat)
                        ys.append(label)
                    if feats:
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

    def _motion_clips(self, folders, tags=None, allow_short=False):
        """딥러닝 학습 묶음. 인식과 같은 1초(15장) 창이고, 끝 묶음은 영상 끝에 맞춥니다.

        돌려주는 묶음은 (태그 번호, 관절, 깊이, 화면 비율, 움직이는 동작인지, 영상 경로) 입니다.
        움직여야 하는 동작이 있으면 태그 끝에 무동작(idle)을 둡니다.
        """
        from gesture.model.motion import stack_points, windows

        points = video_points(self.settings, [path for _name, videos in folders for path in videos])
        window = int(self.settings.seq_window)
        need = int(self.settings.seq_min_frames)
        moving_tags = set(self.settings.moving_tags)
        photo_exts = {item.lower() for item in self.settings.photo_extensions}
        found = []
        used_by_tag = {}
        for name, videos in folders:
            for path in videos:
                frames, aspect = points[path]
                kept = False
                is_photo = path.suffix.lower() in photo_exts
                if is_photo:
                    # 사진 무동작: 손 잡힌 한 장만 넣고, 학습 때 8~15장으로 늘린다.
                    joints, depth = stack_points(frames)
                    if joints is None or joints.shape[0] < 1:
                        continue
                    found.append((name, joints, depth, aspect, False, path))
                    kept = True
                else:
                    for part in windows(frames, window):
                        joints, depth = stack_points(part)
                        if joints is None or (joints.shape[0] < need and not allow_short):
                            continue
                        found.append((name, joints, depth, aspect, name in moving_tags, path))
                        kept = True
                if kept:
                    used_by_tag.setdefault(name, []).append(path.relative_to(self.settings.video_dir).as_posix())
        if tags is None:
            tags = [name for name, _videos in folders if name in used_by_tag]
            if "idle" not in tags and any(name in moving_tags for name in tags):
                tags.append("idle")
        if not found:
            raise RuntimeError("영상에서 손을 찾지 못했습니다. 중앙 박스 안에서 다시 녹화하세요.")
        if len([name for name in tags if name != "idle"]) < 2:
            raise RuntimeError("손이 잡힌 태그가 2종류 이상 있어야 학습할 수 있습니다.")
        index = {name: number for number, name in enumerate(tags)}
        clips = [
            (index[name], joints, depth, aspect, moving, path)
            for name, joints, depth, aspect, moving, path in found
            if name in index
        ]
        videos = [video for name in tags for video in used_by_tag.get(name, [])]
        tag_counts = {name: len(used_by_tag[name]) for name in tags if name in used_by_tag}
        return clips, tags, videos, tag_counts

    def _thresholds_for(self, tags):
        # 등록 화면의 점수는 다음 학습에 스냅샷으로 남는다. 이미 만든 모델은 바뀌지 않는다.
        saved = {tag["id"]: tag.get("threshold") for tag in TagStore(self.settings).load()}
        default = self.settings.conf_threshold
        return {name: saved.get(name, default) for name in tags}

    def _save(self, pipeline, tags, videos, tag_counts, thresholds):
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
                "augmented": True,
                "thresholds": thresholds,
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
            "augmented": True,
            "thresholds": thresholds,
        }
        catalog["models"].append(record)
        catalog["active"] = model_id
        self._write_catalog(catalog)
        return record

    def _save_motion(self, model, videos, tag_counts, thresholds, save_model):
        self.directory.mkdir(parents=True, exist_ok=True)
        catalog = self._read_catalog()
        model_id = self._next_id(catalog)
        filename = f"{model_id}.pt"
        save_model(self.directory / filename, model, thresholds)
        record = {
            "id": model_id,
            "file": filename,
            "kind": "motion",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "video_count": len(videos),
            "videos": videos,
            "tag_counts": tag_counts,
            "augmented": True,
            "thresholds": thresholds,
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


def _split_videos(paths, ratio=0.2):
    """영상 목록에서 시험에 쓸 몫을 뺍니다. 최소 1개는 학습에 남깁니다."""
    items = list(paths)
    random.Random(0).shuffle(items)
    hold = max(1, int(round(len(items) * ratio)))
    if hold >= len(items):
        hold = len(items) - 1
    return items[hold:], items[:hold]


def _forest():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)),
        ]
    )


def _torch_parts():
    try:
        from gesture.model.sequence import save_motion_model, train_motion_model
    except ImportError as exc:
        raise RuntimeError("딥러닝에는 torch가 필요합니다.") from exc
    return train_motion_model, save_motion_model


# 랜덤포레스트는 연속 1초 묶음으로 자른다. 15fps면 15장이 1초이고, 5장마다 다음 묶음을 만든다.
def _windows(frames, length=15, stride=5):
    if len(frames) < length:
        return [frames] if frames else []
    return [frames[start : start + length] for start in range(0, len(frames) - length + 1, stride)]
