"""등록용 영상을 줄여서 태그 폴더에 저장합니다."""

import time

import cv2

from gesture.config import settings as default_settings


class VideoRecorder:
    """중앙 영역을 가로 640, 15fps MJPG로 저장합니다."""

    def __init__(self, settings=None):
        self.settings = settings or default_settings
        self.writer = None
        self.path = None
        self.size = None
        self.started = None
        self._last = 0.0
        self.frames = 0
        self.recording = False

    def start(self, tag):
        folder = self.settings.video_dir / tag
        folder.mkdir(parents=True, exist_ok=True)
        index = self._next_index(folder, tag)
        self.path = folder / f"{tag}_{index:04d}.avi"
        self.writer = None
        self.size = None
        self.frames = 0
        self.started = time.perf_counter()
        self._last = 0.0
        self.recording = True
        self.tag = tag

    def write(self, frame):
        if not self.recording:
            return
        now = time.perf_counter()
        if self.writer is not None and now - self._last < 1.0 / self.settings.video_fps:
            return
        image = resize_max_width(frame, self.settings.video_max_width)
        height, width = image.shape[:2]
        if self.writer is None:
            self.size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(
                str(self.path), fourcc, self.settings.video_fps, self.size
            )
            if not self.writer.isOpened():
                self.recording = False
                self.writer = None
                raise RuntimeError("영상 파일을 만들지 못했습니다.")
        elif (width, height) != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        self.writer.write(image)
        self.frames += 1
        self._last = now

    def stop(self):
        path = self.path
        frames = self.frames
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.recording = False
        self.started = None
        if path is not None and frames == 0 and path.exists():
            path.unlink()
            path = None
        self.path = None
        return path

    def elapsed(self):
        if self.started is None:
            return 0.0
        return time.perf_counter() - self.started

    def _next_index(self, folder, tag):
        nums = []
        for path in folder.glob(f"{tag}_*.avi"):
            try:
                nums.append(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        return (max(nums) + 1) if nums else 1


def resize_max_width(frame, max_width):
    """가로가 max_width를 넘지 않게 줄이고, 코덱용으로 짝수 크기로 맞춥니다."""
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        width = max_width
        height = int(round(height * scale))
    if width % 2:
        width -= 1
    if height % 2:
        height -= 1
    width = max(width, 2)
    height = max(height, 2)
    if (width, height) == (frame.shape[1], frame.shape[0]):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
