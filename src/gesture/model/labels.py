"""영상 태그 수에 맞춰 스무더가 읽을 이름 목록."""


class LabelSettings:
    """학습된 태그 목록을 스무더 설정 형태로 바꿉니다."""

    def __init__(self, tags, settings):
        self.gestures = tuple(tags)
        self.actions = {}
        self.ema_alpha = settings.ema_alpha
        self.hold_frames = settings.hold_frames
        self.conf_threshold = settings.conf_threshold
