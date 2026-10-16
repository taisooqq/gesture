"""웹캠 프레임을 보여주고, 녹화 중이면 중앙 영역만 저장합니다."""

import time
from collections import deque

import cv2
import numpy as np
from PySide6.QtWidgets import QListWidgetItem

from gesture.hand import HandAnalyzer
from gesture.hand_track import HandOverlay
from gesture import pointer
from gesture.recorder import VideoRecorder, open_camera, read_camera
from gesture.smoother import GestureSmoother


class CameraPane:
    """인식 탭과 등록 탭이 같은 카메라 루프를 씁니다."""

    analyzer: HandAnalyzer
    hand_overlay: HandOverlay
    recorder: VideoRecorder
    smoother: GestureSmoother

    def _open_camera(self):
        self.cap = open_camera(self.camera_index)
        if not self.cap.isOpened():
            self.recognize_status.setText("웹캠을 열 수 없습니다.")
            self.register_status.setText("웹캠을 열 수 없습니다.")
            self.cap.release()
            self.cap = None

    def _tick(self):
        if self._closing:
            return
        if self.cap is not None:
            ok, frame = read_camera(self.cap, self.settings)
            if ok:
                self._show(cv2.flip(frame, 1))
        self._play_clip()

    # 인식은 화면 전체를 본다. 중앙 초록 박스는 모니터 전체와 같은 비율이다.
    # 커서를 클릭보다 먼저 옮긴다. 클릭을 먼저 보내면 검지가 내려간 다음 자리를 누르게 된다.
    def _show(self, frame):
        roi, record_box = self.analyzer.crop(frame)
        label, conf = self._predict(frame)
        label, conf = self._gate(label, conf)
        # 인식 탭의 초록 박스는 포인터 영역이다. 등록 탭 네모는 녹화 영역이라 더 작다.
        if self.tabs.currentIndex() == 0:
            box = self.analyzer.center_box(frame, self.settings.pointer_scale)
        else:
            box = record_box
        self._aim(frame, box, label)
        state = self._control(label)
        self._track_log(label, state)
        if getattr(self, "_predict_note", None):
            text = self._predict_note
        elif self.model is None and label == "none":
            text = "제스처: 모델 없음"
        else:
            name = self.settings.gesture_ko.get(label, self._title(label))
            if getattr(self, "_control_note", None):
                text = self._control_note
            elif label in ("none", "idle"):
                text = f"제스처: {name}"
            else:
                text = f"제스처: {name}  {round(conf * 100)}%"
        self.recognize_status.setText(text)

        # 판단에서 이미 찾은 관절로 선을 그린다. 화면용으로 미디어파이프를 다시 돌리지 않는다.
        # 인식 탭은 학습 좌표를 오른쪽 위에 적는다. 등록 탭은 관절 번호를 유지한다.
        points = self.analyzer.landmarks(frame)
        if self.tabs.currentIndex() == 0:
            view = self.hand_overlay.draw(frame, points, numbers=False)
            lines = self.analyzer.tip_coord_labels(frame)
            if lines:
                if view is frame:
                    view = frame.copy()
                self.hand_overlay.write_tip_readout(view, lines)
        else:
            view = self.hand_overlay.draw(frame, points)
        if self.analyzer.hands_error:
            self.register_status.setText(self.analyzer.hands_error)
        x, y, w, h = box
        if self.tabs.currentIndex() == 0:
            color = (60, 190, 80)
        else:
            color = (40, 40, 220) if self.recorder.recording else (60, 190, 80)
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        target = self.recognize_image if self.tabs.currentIndex() == 0 else self.register_image
        self._set_image(target, view)

        if not self.recorder.recording:
            return
        try:
            self.recorder.write(roi)
        except RuntimeError as exc:
            self.recorder.recording = False
            self.record_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.register_status.setText(str(exc))
            return
        self.register_status.setText(
            f"녹화 중 · {self._title(self.tag)} · {self.recorder.elapsed():.1f}초"
        )

    # 인식. 고른 모델이 학습한 숫자 종류로 최근 화면을 바꿔 그 모델에 넣는다.
    def _predict(self, frame):
        self._predict_note = None
        if not hasattr(self, "_recent_frames"):
            self._recent_frames = deque()
        now = time.perf_counter()
        self._recent_frames.append((now, frame))
        while self._recent_frames and now - self._recent_frames[0][0] > 1.0:
            self._recent_frames.popleft()
        if self.model is None:
            self._phase = None
            self.smoother.reset()
            return "none", 0.0
        if not hasattr(self, "_phase"):
            self._phase = None
        # 손이 없으면 동작도 없다. 무동작은 학습 항목이 아니라, 잡고 있던 손을 놓은 상태다.
        contact = self.analyzer.contacts(frame)
        if contact is None:
            self._phase = None
            self._pinch_low = None
            self.smoother.reset()
            return "none", 0.0
        kind = getattr(self, "model_features", None)
        # 윤곽 숫자 8개 모델은 한 장만 본다. 관절 모델은 최근 1초가 모여야 한다.
        single = kind not in HandAnalyzer.WINDOW_KINDS
        if not single and now - self._recent_frames[0][0] < 0.8:
            return "idle", 0.0
        frames = [frame] if single else self._even_frames(15)
        feat = self.analyzer.features_for(frames, kind)
        expected = getattr(self.model, "n_features_in_", None)
        if feat is None or (expected is not None and feat.shape[-1] != expected):
            self.smoother.reset()
            return "idle", 0.0
        # 판단
        proba = self.model.predict_proba(feat.reshape(1, -1))[0]
        ordered = np.zeros(len(self.model_tags), dtype=np.float64)
        for index, cls in enumerate(self.model.classes_):
            ordered[int(cls)] = proba[index]
        # 판단
        label, conf, _action, _fired = self.smoother.update(ordered)
        label = self._swipe_by_facing(frame, label)
        # 우클릭은 모델이 그렇게 말하면 그대로 쓴다. 손가락 거리로 막거나 유지하지 않는다.
        if label == "right_click":
            return label, conf
        # 뒤로가기·앞으로가기만 최근 움직임이 있어야 한다.
        # 새로고침, 스크롤, 좌클릭 누르기·떼기는 자세라서 멈춰 있어도 통과한다.
        if label in self.settings.motion_tags:
            motion = self.analyzer.tip_motion(self._frames_within(self.settings.idle_seconds))
            if motion is None or motion < self.settings.idle_motion:
                return "idle", 0.0
        if label not in ("none", "idle"):
            return label, conf
        return "idle", 0.0

    # 모델이 고른 스와이프를 다른 쪽으로 바꾸지 않는다.
    # 뒤로가기는 모델이 뒤로가기이고 중지 끝 x가 -1 이하일 때만 나간다.
    # 앞으로가기는 모델이 앞으로가기이고 중지 끝 x가 +1 이상일 때만 나간다.
    # 손이 가운데로 돌아오며 반대 면이 보여도, 그때 모델이 말한 동작이 아니면 명령이 나가지 않는다.
    def _swipe_by_facing(self, frame, label):
        """뒤로가기·앞으로가기는 모델 판정과 중지 끝 x가 같이 맞을 때만 통과합니다."""
        if label not in ("back", "forward"):
            return label
        middle_x = self.analyzer.middle_x(frame)
        if middle_x is None:
            return "idle"
        if label == "back" and middle_x <= self.settings.dorsum_x:
            return "back"
        if label == "forward" and middle_x >= self.settings.palm_x:
            return "forward"
        return "idle"

    # 한 번 한 동작이 여러 번 나가지 않게 막는다.
    # 뒤로가기처럼 손을 한 번 저어도 속도가 잠깐 느려지면 무동작이 되었다가
    # 바로 같은 동작으로 다시 보인다. 예전에는 무동작이 되는 즉시
    # "이미 실행했다"는 기억을 지워서, 다음 장면에서 또 실행했다.
    # 이제는 그 동작 이름을 기억해 두고, 무동작이 0.45초 동안 이어지기 전에는
    # 같은 이름을 다시 보내지 않는다. 막힌 장면은 echo 를 돌려서 로그에도 새 줄을 만들지 않는다.
    # 한 번 끝나는 동작은 인식된 뒤 0.45초 동안 다른 동작을 받지 않는다.
    # 스크롤은 자세가 유지되는 동안 휠을 이어 보낸다.
    # 좌클릭은 떼기 모션이 나오기 전까지 유지한다. 떼기는 버튼이 눌려 있을 때만 받는다.
    def _control(self, label):
        """실제 동작 버튼이 켜져 있고 인식 탭일 때만 마우스와 브라우저로 보냅니다."""
        now = time.perf_counter()
        armed = self.tabs.currentIndex() == 0 and self.live_button.isChecked()
        if not armed:
            self._release_click()
            if self.tabs.currentIndex() != 0:
                self._sent_label = None
                self._blocked = None
                return "idle"
            if label in ("none", "idle"):
                self._remember_until_quiet(label, now)
                self._control_note = None
                return "idle"
            if label == getattr(self, "_sent_label", None):
                return None
            if label == getattr(self, "_blocked", None):
                self._idle_since = None
                return "echo"
            self._sent_label = label
            self._blocked = None
            return "off"
        if label == "none":
            self._release_click()
            self._remember_until_quiet(label, now)
            self._control_note = None
            return "idle"
        if label == "idle":
            self._remember_until_quiet(label, now)
            self._control_note = None
            return "idle"
        if label in self._SCROLLS and label == getattr(self, "_sent_label", None):
            return self._repeat_scroll(label, now)
        if label == getattr(self, "_sent_label", None):
            return None
        if label == getattr(self, "_blocked", None):
            self._idle_since = None
            return "echo"
        self._blocked = None
        self._sent_label = label
        if label in self._MOMENTARY:
            self._lock_until = now + self.settings.refire_seconds
            self._lock_label = label
        if label == "left_click_on":
            ok = pointer.left_down()
            if ok:
                self._click_held = True
        elif label == "left_click_off":
            ok = pointer.left_up()
            self._click_held = False
        elif label == "right_click":
            ok = pointer.right_click()
        elif label in ("back", "forward", "refresh"):
            ok = pointer.browser(label)
        elif label == "screenshot":
            ok = pointer.screenshot()
        elif label in self._SCROLLS:
            ok = pointer.scroll(self._SCROLLS[label])
            if ok:
                self._scroll_at = now
        else:
            return "off"
        if ok:
            self._control_note = None
            return "ok"
        self._control_note = "손쉬운 사용 권한이 필요합니다"
        return "fail"

    _SCROLLS = {"scroll_up": 1, "scroll_down": -1}
    _MOMENTARY = ("back", "forward", "refresh", "right_click", "screenshot")
    _TIMED = {"left_click_on", "scroll_up", "scroll_down"}
    _NO_REFIRE = {"left_click_on", "left_click_off", "scroll_up", "scroll_down"}

    def _gate(self, label, conf):
        """모델 결과를 유지 규칙에 맞게 다시 고릅니다."""
        now = time.perf_counter()
        if label == "left_click_on":
            self._click_conf = conf
        if getattr(self, "_click_held", False):
            if label == "none":
                return label, conf
            if label == "left_click_off":
                return label, conf
            return "left_click_on", getattr(self, "_click_conf", conf)
        if label == "left_click_off":
            return "idle", 0.0
        if now < getattr(self, "_lock_until", 0.0) and label != getattr(self, "_lock_label", None):
            return "idle", 0.0
        return label, conf

    def _release_click(self):
        if getattr(self, "_click_held", False):
            pointer.left_up()
            self._click_held = False

    def _repeat_scroll(self, label, now):
        """같은 스크롤 자세가 유지되면 휠을 이어서 보냅니다."""
        last = getattr(self, "_scroll_at", 0.0)
        if now - last < self.settings.scroll_interval:
            return None
        self._scroll_at = now
        ok = pointer.scroll(self._SCROLLS[label])
        if ok:
            self._control_note = None
            return "ok"
        self._control_note = "손쉬운 사용 권한이 필요합니다"
        return "fail"

    # 동작이 무동작으로 바뀌면 그 이름을 _blocked 에 남긴다.
    # 무동작이 refire_seconds 만큼 이어지면 기억을 지워서 다음 실행을 허용한다.
    # 그 전에 같은 동작이 다시 보이면 _control 이 실행을 막는다.
    # 좌클릭 누르기·떼기와 스크롤은 여기에 넣지 않는다. 더블클릭과 연속 스크롤이 0.45초를 기다리지 않게 하려고다.
    def _remember_until_quiet(self, label, now):
        """동작이 끝나면 그 이름을 남기고, 무동작이 충분히 길면 다음 실행을 허용합니다."""
        if getattr(self, "_blocked", None) is None and self._sent_label not in (
            None,
            "none",
            "idle",
            *self._NO_REFIRE,
        ):
            self._blocked = self._sent_label
            self._idle_since = now
        elif getattr(self, "_blocked", None) and getattr(self, "_idle_since", None) is None:
            self._idle_since = now
        self._sent_label = label
        if (
            getattr(self, "_blocked", None)
            and getattr(self, "_idle_since", None) is not None
            and now - self._idle_since >= self.settings.refire_seconds
        ):
            self._blocked = None
            self._idle_since = None

    # 오른쪽 목록은 동작이 바뀔 때마다 한 줄을 만든다.
    # 같은 동작이 무동작으로 깜빡여 다시 보여도, echo 이면 줄을 만들지 않는다.
    # 한 번 손을 젓고 페이지가 한 번만 넘어갔는데 기록이 여러 줄이 되지 않게 하려고다.
    def _track_log(self, label, state):
        """오른쪽 목록에 동작 이름, 유지 시간, 실행 여부를 남깁니다."""
        if state == "echo":
            return
        now = time.perf_counter()
        showing = label not in ("none", "idle") and state != "idle"
        current = getattr(self, "_log_label", None)
        if showing and label == current:
            self._paint_log(now)
            return
        if current not in (None, "none", "idle"):
            self._paint_log(now)
            self._log_label = None
        if not showing:
            return
        self._log_label = label
        self._log_started = now
        self._log_state = state if state in ("ok", "fail", "off") else "off"
        self._log_clock = time.strftime("%H:%M:%S")
        item = QListWidgetItem(self._log_text(0.0))
        self.gesture_log.insertItem(0, item)
        self._log_item = item
        while self.gesture_log.count() > 40:
            self.gesture_log.takeItem(self.gesture_log.count() - 1)

    def _paint_log(self, now):
        item = getattr(self, "_log_item", None)
        if item is None:
            return
        item.setText(self._log_text(now - self._log_started))

    def _log_text(self, seconds):
        name = self.settings.gesture_ko.get(self._log_label, self._title(self._log_label))
        mark = {"ok": "실행", "fail": "실패", "off": "인식"}.get(self._log_state, "인식")
        if self._log_label not in self._TIMED:
            return f"{self._log_clock}  {name}  {mark}"
        return f"{self._log_clock}  {name}  {seconds:.1f}초  {mark}"

    # 초록 박스 안 검지 끝은 모니터의 같은 비율 위치다. 박스 왼쪽 위는 화면 왼쪽 위다.
    # 손이 없으면 커서를 더 옮기지 않는다. 트랙패드로 마우스를 움직이게 하려고다.
    # 좌클릭 누르기 순간 검지 끝을 고정한다. 엄지를 접으며 검지가 움직이면
    # 클릭 위치가 밀리기 때문이다. 떼기 전까지는 손목이 움직인 만큼만 옮긴다.
    # 이 함수는 클릭을 보내기 전에 호출된다. 클릭이 기억해 둔 자리에서 나가게 하려고다.
    def _aim(self, frame, box, label):
        """초록 박스 안 검지 끝을 모니터의 같은 위치로 옮깁니다."""
        if self.tabs.currentIndex() != 0 or not self.live_button.isChecked():
            self._aim_lock = None
            pointer.release_cursor()
            return
        if not pointer.trusted():
            self._control_note = "손쉬운 사용 권한이 필요합니다"
            return
        points = self.analyzer.pointing(frame)
        if points is None:
            self._aim_lock = None
            pointer.release_cursor()
            return
        pointer.arm_cursor()
        tip_x, tip_y, wrist_x, wrist_y = points
        # 가리킬 때는 검지 끝이 모니터의 그 자리다.
        # 누르기 모션이거나 버튼이 이미 눌린 동안만 그 자리의 검지를 고정하고 손목만큼 옮긴다.
        locked = label == "left_click_on" or getattr(self, "_click_held", False)
        x, y, width, height = box
        if locked:
            if getattr(self, "_aim_lock", None) is None:
                self._aim_lock = (tip_x, tip_y, wrist_x, wrist_y)
            lock_x, lock_y, lock_wrist_x, lock_wrist_y = self._aim_lock
            place_x = lock_x + (wrist_x - lock_wrist_x)
            place_y = lock_y + (wrist_y - lock_wrist_y)
        else:
            self._aim_lock = None
            place_x, place_y = tip_x, tip_y
        rel_x = min(1.0, max(0.0, (place_x - x) / width))
        rel_y = min(1.0, max(0.0, (place_y - y) / height))
        if not pointer.move_ratio(rel_x, rel_y):
            self._control_note = "손쉬운 사용 권한이 필요합니다"

    def _frames_within(self, seconds):
        """최근 몇 초 안의 화면만 고릅니다."""
        if not self._recent_frames:
            return []
        now = self._recent_frames[-1][0]
        return [frame for stamp, frame in self._recent_frames if now - stamp <= seconds]

    def _even_frames(self, count):
        """최근 1초에 모인 화면을 학습 묶음과 같은 장 수로 고릅니다."""
        frames = [frame for _stamp, frame in self._recent_frames]
        if len(frames) <= count:
            return frames
        indexes = np.linspace(0, len(frames) - 1, num=count, dtype=int)
        return [frames[int(index)] for index in indexes]
