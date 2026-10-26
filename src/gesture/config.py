"""경로, 제스처 이름, 판정 기준."""

from pathlib import Path


def clamp_threshold(value, default=0.65):
    """인식 점수를 0.01~0.99로 맞춥니다. 1이면 거의 통과하지 않습니다."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return min(0.99, max(0.01, number))


class Settings:
    """프로젝트 전역 설정. 제스처나 명령을 바꿀 때는 이 클래스만 수정합니다."""

    def __init__(self):
        self.root = Path(__file__).resolve().parents[2]
        self.gestures = ("fist", "palm", "scissors", "one", "thumb")
        self.gesture_ko = {
            "none": "손 없음",
            "idle": "무동작",
            "fist": "주먹",
            "palm": "보",
            "scissors": "가위",
            "one": "손가락 1",
            "thumb": "엄지",
            "back_idle": "뒤로가기후돌아오기",
            "forward_idle": "앞으로가기후돌아오기",
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
        # 태그에 점수가 없을 때 쓰는 기본값. 동작마다 다른 점수는 태그 threshold 다.
        self.conf_threshold = 0.65
        # 손가락 거리로 유지하지 않는다. 좌클릭은 누르기·떼기 모션이 각각 명령을 보낸다.
        self.hold_tags = ()
        # 손을 한 번 저어야 하는 동작. 자세로 바뀐 새로고침과 스크롤, 클릭은 여기 없다.
        self.motion_tags = ("back", "forward")
        # 스크롤 자세가 유지되는 동안 휠을 보내는 간격.
        self.scroll_interval = 0.08
        # 좌클릭 누르기만 최근 이 프레임으로 본다. 15fps면 10장은 약 0.67초다.
        self.press_frames = 10
        # 좌클릭 누르기·짧은 떼기 뒤 이 시간 동안은 클릭 자리에 커서를 고정한다.
        # 같은 자리에서 실제 클릭이 두 번 나가 OS 더블클릭이 되게 하려는 시간이다.
        self.press_still_seconds = 0.4
        # 더블의 마지막 떼기 뒤에도 이 시간만큼 자리를 유지한다.
        self.press_release_still_seconds = 0.1
        # 좌클릭을 누른 뒤 떼기만 최근 이 시간으로 본다.
        # 1초 창에는 접힌 자세가 남아 떼기가 늦다.
        self.release_seconds = 0.5
        # 손 크기(손목~중지 뿌리)로 나눈 끝 거리. 이보다 가까우면 잡고, 멀어지면 놓는다.
        self.pinch_on = 0.42
        self.pinch_off = 0.62
        # 좌클릭 중 가장 붙었던 거리보다 이만큼 벌어지면 손을 뗀 것으로 본다.
        # 중지 10번은 떼어도 0.62보다 가까운 경우가 많아서, 절대 거리만으로는 클릭이 안 끝난다.
        self.pinch_release = 0.2
        # 한 번 움직이는 동작은 최근 0.3초가 이보다 크게 움직일 때만 그 동작이다.
        self.idle_seconds = 0.3
        self.idle_motion = 0.12
        # 같은 제스처가 무동작으로 깜빡여도, 이 시간 동안 쉬기 전에는 다시 실행하지 않는다.
        self.refire_seconds = 0.45
        # 인식 화면 좌표 표시용. 중지 끝 x가 이보다 왼쪽이면 손등, 오른쪽이면 손바닥으로 적는다.
        self.dorsum_x = -1.0
        self.palm_x = 1.0
        self.min_hand_ratio = 0.03
        # 등록 녹화만 이 비율의 정중앙을 남긴다. 얼굴이 영상에 들어가지 않게 하려는 영역이다.
        self.roi_scale = 0.55
        # 인식 탭의 초록 박스. 화면 위에 붙고, 이 안 검지 끝이 모니터 전체다.
        self.pointer_scale = 0.8
        self.data_dir = self.root / "data"
        self.video_dir = self.data_dir / "videos"
        # 영상마다 한 번 뽑은 손 좌표. 영상이 바뀌면 다시 뽑는다.
        self.landmark_dir = self.data_dir / "landmarks"
        # 손 좌표를 뽑는 프로세스 수. None 이면 코어 수에서 하나를 뺀다.
        self.landmark_workers = None
        self.tags_path = self.data_dir / "tags.json"
        self.video_max_width = 640
        self.video_fps = 15
        # 카메라 화면 비율. 가로 1과 세로 1의 길이가 달라지면 손 좌표가 찌그러진다.
        self.camera_aspect_w = 16
        self.camera_aspect_h = 9
        # 이미 16:9가 아니면 앞에서부터 요청한다. 맥 FaceTime은 1920×1080이라 그대로 둔다.
        self.camera_sizes = ((1920, 1080), (1280, 720))
        # 0이 가벼운 손 모델이다. 1은 더 정확하고 더 느리다.
        self.hand_complexity = 1
        # 모델 8부터 학습 좌표만 불린다. 위치와 크기는 이미 빠져 있어 각도, 속도, 흔들림만 넣는다.
        self.augment_rotations = (-18, -9, 9, 18)
        self.augment_speeds = (0.65, 1.4)
        self.augment_jitter = 0.02
        self.augment_jitter_copies = 2
        # 양수는 위를 늘리고 아래를 줄인다. 음수는 위를 줄이고 아래를 늘린다.
        self.augment_trapezoids = (0.05, 0.10, -0.05, -0.10)
        # 딥러닝은 인식과 같이 최근 1초를 15fps로 본다. 손이 이보다 적게 잡히면 판단하지 않는다.
        self.seq_window = 15
        self.seq_min_frames = 3
        self.seq_epochs = 50
        # 동작 점수가 무동작보다 이만큼 더 높을 때만 그 동작으로 본다. 오인을 줄인다.
        self.idle_margin = 0.15
        # 움직여야 그 동작인 태그. 처음·끝 자세로 멈춘 손은 딥러닝이 무동작으로 배운다.
        # 스크롤(hold_pose_tags)은 무동작 합성에서 빼고, 우클릭·새로고침·스크린샷의 멈춘 손만 무동작이다.
        # back_idle·forward_idle 은 스와이프 후 손이 돌아오는 움직임이다. 브라우저 명령은 안 보낸다.
        self.moving_tags = (
            "left_click_on",
            "left_click_off",
            "back",
            "forward",
            "back_idle",
            "forward_idle",
        )
        self.hold_pose_tags = ("scroll_up", "scroll_down")
        # 인식만 하고 마우스·키는 안 보내는 태그. 반대 스와이프 오인을 먹는다.
        self.idle_gesture_tags = ("back_idle", "forward_idle")
        self.train_tag_ids = (
            "left_click_on",
            "left_click_off",
            "right_click",
            "back",
            "forward",
            "back_idle",
            "forward_idle",
            "refresh",
            "screenshot",
            "scroll_up",
            "scroll_down",
        )
        self.video_tags = (
            ("left_click_on", "마우스 좌클릭 누르기", "검지를 핀상태에서 엄지 접기"),
            ("left_click_off", "마우스 좌클릭 떼기", "접은 엄지 펴기"),
            ("right_click", "우클릭", "엄지와 검지를 제외한 나머지 손가락이 맞닿음"),
            ("back", "뒤로가기", "손바닥을 세우고 오른쪽에서 왼쪽으로 움직임"),
            ("forward", "앞으로가기", "손바닥을 세우고 왼쪽에서 오른쪽으로 움직임"),
            ("back_idle", "뒤로가기후돌아오기", "뒤로가기 후 손을 가운데로 되돌림"),
            ("forward_idle", "앞으로가기후돌아오기", "앞으로가기 후 손을 가운데로 되돌림"),
            ("refresh", "새로고침", "새끼손가락 올림"),
            ("screenshot", "스크린샷", "중지올리기"),
            ("scroll_up", "위로 스크롤", "따봉"),
            ("scroll_down", "아래로 스크롤", "로키 따봉"),
        )
        self.model_path = self.root / "models" / "gesture.joblib"


settings = Settings()
