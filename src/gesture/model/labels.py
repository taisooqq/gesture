"""영상 태그 수에 맞춰 스무더가 읽을 이름 목록."""

from gesture.config import clamp_threshold


class LabelSettings:
    """학습된 태그 목록을 스무더 설정 형태로 바꿉니다."""

    def __init__(self, tags, settings, thresholds=None):
        self.gestures = tuple(tags)
        self.actions = {}
        self.ema_alpha = settings.ema_alpha
        self.hold_frames = settings.hold_frames
        self.conf_threshold = settings.conf_threshold
        # 학습 때 저장한 동작별 점수. 없는 예전 모델은 모두 기본 점수를 쓴다.
        saved = thresholds or {}
        default = settings.conf_threshold
        self.thresholds = tuple(
            clamp_threshold(saved.get(tag, default), default) for tag in self.gestures
        )
