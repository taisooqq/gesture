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
        self.data_dir = self.root / "data"
        self.video_dir = self.data_dir / "videos"
        self.tags_path = self.data_dir / "tags.json"
        self.video_max_width = 640
        self.video_fps = 15
        self.video_tags = (
            ("pointer", "마우스포인터 움직임", "검지손가락의 방향"),
            ("left_click", "마우스 좌클릭", "검지와 엄지가 맞닿음"),
            ("right_click", "우클릭", "엄지와 중지가 맞닿음"),
            ("back", "뒤로가기", "검지, 중지를 펴고 오른쪽에서 왼쪽으로"),
            ("forward", "앞으로가기", "검지, 중지를 펴고 왼쪽에서 오른쪽으로"),
            ("refresh", "새로고침", "검지를 돌림"),
            ("drag", "드래그", "좌클릭을 유지"),
            ("drop", "드롭", "손을 놓음"),
            ("idle", "무동작", "주먹"),
        )
        self.model_path = self.root / "models" / "gesture.joblib"


settings = Settings()
