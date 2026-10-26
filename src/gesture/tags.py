"""등록 태그를 JSON 파일에 읽고 씁니다."""

import json
import re

from gesture.config import clamp_threshold
from gesture.config import settings as default_settings

_FOLDER_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class TagStore:
    """data/tags.json. 없으면 기본 태그를 만들어 저장합니다."""

    def __init__(self, settings=None):
        self.settings = settings or default_settings
        self.path = self.settings.tags_path

    def load(self):
        if not self.path.exists():
            tags = [
                {
                    "id": tag_id,
                    "title": title,
                    "detail": detail,
                    "train": tag_id in self.settings.train_tag_ids,
                    "threshold": self.settings.conf_threshold,
                    "photo": tag_id in self.settings.photo_tags,
                }
                for tag_id, title, detail in self.settings.video_tags
            ]
            self.save(tags)
            return tags
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        tags = []
        changed = False
        defaults_photo = set(self.settings.photo_tags)
        for item in raw.get("tags", []):
            if not isinstance(item, dict):
                continue
            tag_id = str(item.get("id", "")).strip()
            title = str(item.get("title", "")).strip()
            detail = str(item.get("detail", "")).strip()
            if not tag_id or not title:
                continue
            if isinstance(item.get("train"), bool):
                train = item["train"]
            else:
                train = tag_id in self.settings.train_tag_ids
                changed = True
            threshold = clamp_threshold(item.get("threshold", self.settings.conf_threshold))
            if item.get("threshold") != threshold:
                changed = True
            if isinstance(item.get("photo"), bool):
                photo = item["photo"]
            else:
                photo = tag_id in defaults_photo
                changed = True
            tags.append(
                {
                    "id": tag_id,
                    "title": title,
                    "detail": detail,
                    "train": train,
                    "threshold": threshold,
                    "photo": photo,
                }
            )
        tags, filled = self._ensure_defaults(tags)
        if changed or filled:
            self.save(tags)
        return tags

    def _ensure_defaults(self, tags):
        """기본 태그가 JSON에 없으면 붙입니다. 무동작(사진) 태그를 빠뜨리지 않으려고다."""
        have = {tag["id"] for tag in tags}
        filled = False
        for tag_id, title, detail in self.settings.video_tags:
            if tag_id in have:
                continue
            tags.append(
                {
                    "id": tag_id,
                    "title": title,
                    "detail": detail,
                    "train": tag_id in self.settings.train_tag_ids,
                    "threshold": self.settings.conf_threshold,
                    "photo": tag_id in self.settings.photo_tags,
                }
            )
            filled = True
        return tags, filled

    def save(self, tags):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"tags": tags}, ensure_ascii=False, indent=2) + "\n"
        self.path.write_text(text, encoding="utf-8")

    def train_ids(self):
        return [tag["id"] for tag in self.load() if tag.get("train")]

    def photo_ids(self):
        """사진(무동작)으로 저장하는 태그 id 목록."""
        return {tag["id"] for tag in self.load() if tag.get("photo")}

    def add(self, title, detail, train=False, threshold=None, folder_id=None, photo=False):
        tags = self.load()
        tag_id = self._normalize_id(folder_id)
        if not tag_id:
            tag_id = self._new_id(tags)
        if any(tag["id"] == tag_id for tag in tags):
            raise ValueError(f"폴더명 '{tag_id}' 이(가) 이미 있습니다.")
        tag = {
            "id": tag_id,
            "title": title.strip(),
            "detail": detail.strip(),
            "train": bool(train),
            "threshold": self._threshold(threshold),
            "photo": bool(photo),
        }
        tags.append(tag)
        self.save(tags)
        return tag

    def update(self, tag_id, title, detail, train=False, threshold=None, photo=None):
        tags = self.load()
        for tag in tags:
            if tag["id"] == tag_id:
                tag["title"] = title.strip()
                tag["detail"] = detail.strip()
                tag["train"] = bool(train)
                tag["threshold"] = self._threshold(threshold)
                if photo is not None:
                    tag["photo"] = bool(photo)
                break
        self.save(tags)

    def _threshold(self, value):
        if value is None:
            return self.settings.conf_threshold
        return clamp_threshold(value, self.settings.conf_threshold)

    def delete(self, tag_id):
        tags = [tag for tag in self.load() if tag["id"] != tag_id]
        self.save(tags)

    def _normalize_id(self, value):
        """폴더명: 영문 소문자로 시작하고, 소문자·숫자·_ 만 허용합니다."""
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        if not _FOLDER_ID.fullmatch(text):
            raise ValueError("폴더명은 영문 소문자로 시작하고, 소문자·숫자·_ 만 쓸 수 있습니다.")
        return text

    def _new_id(self, tags):
        numbers = []
        for tag in tags:
            match = re.fullmatch(r"tag_(\d+)", tag["id"])
            if match:
                numbers.append(int(match.group(1)))
        return f"tag_{max(numbers, default=0) + 1:02d}"
