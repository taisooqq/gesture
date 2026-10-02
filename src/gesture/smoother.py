"""연속 프레임의 확률을 모아, 흔들린 판정은 명령으로 보내지 않습니다."""

import numpy as np

from gesture.config import settings as default_settings


class GestureSmoother:
    """EMA로 확률을 부드럽게 하고, 같은 판정이 유지될 때만 명령을 한 번 냅니다."""

    def __init__(self, settings=None):
        self.settings = settings or default_settings
        n = len(self.settings.gestures)
        self.ema = np.zeros(n, dtype=np.float64)
        self.label = "none"
        self.count = 0
        self.sent = None

    def update(self, proba):
        alpha = self.settings.ema_alpha
        self.ema = alpha * proba + (1.0 - alpha) * self.ema
        idx = int(np.argmax(self.ema))
        conf = float(self.ema[idx])
        label = self.settings.gestures[idx] if conf >= self.settings.conf_threshold else "none"
        if label == self.label:
            self.count += 1
        else:
            self.label = label
            self.count = 1
            self.sent = None

        action = self.settings.actions.get(label, "none")
        ready = (
            label != "none"
            and action != "none"
            and self.count >= self.settings.hold_frames
            and self.sent != label
        )
        if ready:
            self.sent = label
        return label, conf, action if ready else "none", ready

    def reset(self):
        self.ema[:] = 0
        self.label = "none"
        self.count = 0
        self.sent = None
