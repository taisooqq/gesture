"""등록 태그를 JSON 파일에 읽고 씁니다."""

import json
import re

from gesture.config import settings as default_settings


class TagStore:
    """data/tags.json. 없으면 기본 태그 9개를 만들어 저장합니다."""

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
                }
                for tag_id, title, detail in self.settings.video_tags
            ]
            self.save(tags)
            return tags
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        tags = []
        changed = False
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
            tags.append({"id": tag_id, "title": title, "detail": detail, "train": train})
        if changed:
            self.save(tags)
        return tags

    def save(self, tags):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"tags": tags}, ensure_ascii=False, indent=2) + "\n"
        self.path.write_text(text, encoding="utf-8")

    def train_ids(self):
        return [tag["id"] for tag in self.load() if tag.get("train")]

    def add(self, title, detail, train=False):
        tags = self.load()
        tag = {
            "id": self._new_id(tags),
            "title": title.strip(),
            "detail": detail.strip(),
            "train": bool(train),
        }
        tags.append(tag)
        self.save(tags)
        return tag

    def update(self, tag_id, title, detail, train=False):
        tags = self.load()
        for tag in tags:
            if tag["id"] == tag_id:
                tag["title"] = title.strip()
                tag["detail"] = detail.strip()
                tag["train"] = bool(train)
                break
        self.save(tags)

    def delete(self, tag_id):
        tags = [tag for tag in self.load() if tag["id"] != tag_id]
        self.save(tags)

    def _new_id(self, tags):
        numbers = []
        for tag in tags:
            match = re.fullmatch(r"tag_(\d+)", tag["id"])
            if match:
                numbers.append(int(match.group(1)))
        return f"tag_{max(numbers, default=0) + 1:02d}"
