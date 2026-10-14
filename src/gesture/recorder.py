"""등록용 영상을 줄여서 태그 폴더에 저장합니다."""

import sys
import time
import uuid

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

    # 학습 영상 1. data/videos/태그/태그_a1b2c3.avi 처럼 짧은 임의 이름으로 경로만 잡는다.
    # 번호를 쓰지 않는다. 다른 컴퓨터 영상을 합칠 때 이름이 겹치지 않게 하려는 것이다.
    # 작성기는 첫 프레임이 들어올 때 연다.
    def start(self, tag):
        folder = self.settings.video_dir / tag
        folder.mkdir(parents=True, exist_ok=True)
        self.path = self._new_path(folder, tag)
        self.writer = None
        self.size = None
        self.frames = 0
        self.started = time.perf_counter()
        self._last = 0.0
        self.recording = True
        self.tag = tag

    # 학습 영상 3. 15fps보다 빠른 장은 버린다.
    # 가로는 640 이하, 짝수 크기로 줄인 뒤 MJPG avi에 쓴다. 크기는 첫 장으로 고정한다.
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
            self.writer = open_writer(self.path, self.settings.video_fps, self.size)
            if not self.writer.isOpened():
                self.recording = False
                self.writer = None
                raise RuntimeError("영상 파일을 만들지 못했습니다.")
        elif (width, height) != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        self.writer.write(image)
        self.frames += 1
        self._last = now

    # 학습 영상 4. 파일을 닫는다. 한 장도 없으면 빈 파일은 지운다.
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

    def _new_path(self, folder, tag):
        while True:
            token = uuid.uuid4().hex[:6]
            path = folder / f"{tag}_{token}.avi"
            if not path.exists():
                return path


def open_camera(index, settings=None):
    """웹캠을 연다. 윈도우 기본 방식은 카메라를 못 여는 경우가 많아 DirectShow를 먼저 쓴다."""
    settings = settings or default_settings
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            _select_aspect(cap, settings)
            return cap
        cap.release()
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        _select_aspect(cap, settings)
    return cap


def read_camera(cap, settings=None):
    """한 장을 읽고, 설정한 화면 비율이 아니면 가운데를 잘라 맞춥니다."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None
    return True, fit_aspect(frame, settings)


def fit_aspect(frame, settings=None):
    """가로·세로가 camera_aspect 가 아니면 가운데를 잘라 그 비율로 맞춥니다."""
    settings = settings or default_settings
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return frame
    target = settings.camera_aspect_w / settings.camera_aspect_h
    if abs(width / height - target) <= 0.01:
        return frame
    if width / height > target:
        new_w = int(height * target)
        new_w -= new_w % 2
        new_w = max(new_w, 2)
        x = max(0, (width - new_w) // 2)
        return frame[:, x : x + new_w]
    new_h = int(width / target)
    new_h -= new_h % 2
    new_h = max(new_h, 2)
    y = max(0, (height - new_h) // 2)
    return frame[y : y + new_h, :]


def _select_aspect(cap, settings):
    """이미 16:9면 해상도를 바꾸지 않습니다. 아니면 16:9 크기부터 요청합니다."""
    width, height = _camera_size(cap)
    target = settings.camera_aspect_w / settings.camera_aspect_h
    if _matches_aspect(width, height, target):
        return
    for ask_w, ask_h in settings.camera_sizes:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ask_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ask_h)
        width, height = _camera_size(cap)
        if _matches_aspect(width, height, target):
            return


def _camera_size(cap):
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame.shape[1], frame.shape[0]
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def _matches_aspect(width, height, target):
    if width <= 0 or height <= 0:
        return False
    return abs(width / height - target) <= 0.01


def open_video(path):
    """저장 영상을 연다. 윈도우 기본 방식은 MJPG avi를 넘기며 읽지 못하는 경우가 있다."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(str(path))


def open_writer(path, fps, size):
    """MJPG avi 작성기. 맥과 윈도우가 같은 파일을 만들도록 OpenCV 자체 저장을 쓴다."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), cv2.CAP_OPENCV_MJPEG, fourcc, fps, size)
    if writer.isOpened():
        return writer
    writer.release()
    return cv2.VideoWriter(str(path), fourcc, fps, size)


# 학습 영상 3. 가로가 640을 넘으면 비율을 유지해 줄이고, 가로·세로는 짝수로 맞춘다.
def resize_max_width(frame, max_width):
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
