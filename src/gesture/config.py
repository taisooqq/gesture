"""경로, 제스처 이름, 판정 기준."""

from pathlib import Path


class Settings:
    """프로젝트 전역 설정. 제스처나 명령을 바꿀 때는 이 클래스만 수정합니다."""

    def __init__(self):
        self.root = Path(__file__).resolve().parents[2]
        self.gestures = ("fist", "palm", "scissors", "one", "thumb")
        self.gesture_ko = {
            "none": "손 없음",
            "fist": "주먹",
            "palm": "보",
            "scissors": "가위",
            "one": "손가락 1",
            "thumb": "엄지",
        }
        self.actions = {
            "fist": "next",
            "palm": "prev",
            "scissors": "none",
            "one": "none",
            "thumb": "none",
        }
        self.ema_alpha = 0.35
        self.hold_frames = 8
        self.conf_threshold = 0.65
        self.min_hand_ratio = 0.03
        self.roi_scale = 0.55
        self.web_port = 8765
        self.data_dir = self.root / "data"
        self.model_path = self.root / "models" / "gesture.joblib"
        self.web_dir = self.root / "web"


settings = Settings()
