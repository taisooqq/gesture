"""피부색으로 손을 찾고, 형태를 숫자 특징으로 바꿉니다."""

import cv2
import numpy as np

from gesture.config import settings as default_settings


class HandAnalyzer:
    """중앙 영역 자르기, 피부색 마스크, 형태 특징 추출."""

    # 손 윤곽 하나에서 재는 숫자 8개의 이름.
    # 작업 순서가 아니라, 한 장의 사진이 모델에 들어갈 때 늘어서는 순서다.
    # [면적비, 가로세로비, solidity, 원형도, 손가락 골 개수, hu1, hu2, hu3]
    FEATURE_NAMES = (
        "area_ratio",
        "aspect",
        "solidity",
        "circularity",
        "defects",
        "hu1",
        "hu2",
        "hu3",
    )

    def __init__(self, settings=None):
        self.settings = settings or default_settings

    def crop(self, frame):
        """화면 중앙 박스. 얼굴이 들어가지 않게 손만 남깁니다."""
        h, w = frame.shape[:2]
        # 화면의 약 55%만 남긴다. 얼굴이 저장·인식에 들어가지 않게 하려는 영역이다.
        rw = int(w * self.settings.roi_scale)
        rh = int(h * self.settings.roi_scale)
        x1 = (w - rw) // 2
        y1 = (h - rh) // 2
        return frame[y1 : y1 + rh, x1 : x1 + rw].copy(), (x1, y1, rw, rh)

    def skin_mask(self, bgr):
        """1·2단계. 자른 컬러 화면을 흑백 손 도장(마스크)으로 바꿉니다."""
        # 1. 피부색만 흰색으로 남김.
        # 한 점은 파랑·초록·빨강으로 저장돼 있다. 조명이 밝아지면 세 값이 같이 커져
        # 피부인지 구분하기 어렵다. YCrCb는 밝기(Y)와 색(Cr, Cb)을 나눠 적는다.
        # 피부는 밝기가 달라도 Cr, Cb가 비슷한 구간에 모인다.
        # Y는 거의 제한하지 않고 Cr 133~173, Cb 77~127만 흰색으로 칠한다.
        # 구간 밖은 검정이다. 컬러 사진은 사라지고 손 모양의 흑백 도장만 남는다.
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        # 2. 작은 점은 지우고, 손 안의 구멍은 메운다.
        # 벽의 작은 반사가 흰 점으로 남거나, 그림자가 손바닥에 검정 구멍을 만든다.
        # 동그란 작은 지우개로 두 번 다듬는다.
        # 열기: 지우개보다 작은 흰 점만 지운다. 손처럼 큰 덩어리는 남는다.
        # 닫기: 손 안의 작은 검정 구멍을 흰색으로 채운다.
        # 손가락 사이처럼 원래 뚫린 큰 틈까지 메우지는 않는다.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def extract(self, bgr):
        """3·4단계. 도장 테두리를 손으로 보고, 그 테두리에서 숫자 8개를 잽니다."""
        # 3. 흰색 덩어리의 테두리를 따라가 손으로 본다.
        # 흰색과 검정이 만나는 경계의 좌표 목록이 윤곽이다.
        # 덩어리가 여러 개면 넓이가 가장 큰 테두리를 쓴다.
        # 그 넓이가 자른 화면의 3%보다 작으면 손이 없다고 보고 여기서 끝낸다.
        mask = self.skin_mask(bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape[:2]
        frame_area = float(h * w)
        if not contours:
            return None, mask

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < frame_area * self.settings.min_hand_ratio:
            return None, mask

        # 4. 그 테두리에서 숫자 8개를 계산한다. 모델은 나중에 이 한 줄만 본다.
        # 면적: 테두리 안쪽 넓이를 자른 화면 전체로 나눈 비율.
        # 가로·세로: 테두리를 감싼 직사각형의 가로를 세로로 나눈 값.
        # 볼록 넓이: 테두리 바깥에 고무줄을 씌운 안쪽 넓이. 손가락 홈으로는 들어가지 않는다.
        #   실제 면적 / 고무줄 안 넓이 = solidity. 주먹은 1에 가깝고 보자기는 더 작다.
        # 둘레: 테두리를 한 바퀴 간 길이. 4*π*면적/둘레² 가 원형도이고 원은 1에 가깝다.
        # 홈 개수와 형태값 3개는 아래 4번 보조 함수에서 구한다.
        _x, _y, bw, bh = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) + 1e-6
        peri = float(cv2.arcLength(contour, True)) + 1e-6
        feat = np.array(
            [
                area / frame_area,
                bw / (bh + 1e-6),
                area / hull_area,
                4.0 * np.pi * area / (peri * peri),
                float(self._defect_count(contour)),
                *self._hu3(contour),
            ],
            dtype=np.float32,
        )
        return feat, mask

    def _defect_count(self, contour):
        """4단계. 손가락 사이처럼 깊게 들어간 홈의 개수."""
        # 고무줄(볼록 껍질)과 실제 테두리 사이로 깊게 들어간 곳을 센다.
        # 보자기는 손가락 사이에 홈이 여러 개이고, 주먹은 거의 없다.
        # 얕은 울퉁불퉁함은 빼고 깊이가 18보다 큰 홈만 개수에 넣는다.
        if len(contour) < 4:
            return 0
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            return 0
        try:
            defects = cv2.convexityDefects(contour, hull_idx)
        except cv2.error:
            return 0
        if defects is None:
            return 0
        defects = np.asarray(defects)
        if defects.ndim == 3:
            defects = defects[:, 0, :]
        count = 0
        for row in defects:
            # depth는 고정소수라 256으로 나눈다. 얕은 굴곡은 손가락으로 세지 않는다.
            if float(row[3]) / 256.0 > 18:
                count += 1
        return count

    def _hu3(self, contour):
        """4단계. 테두리 모양을 요약하는 형태값 3개."""
        # Hu 모멘트는 손이 화면 안에서 조금 움직이거나 돌아가도 크게 안 변한다.
        # 7개 중 앞 3개만 쓰고, 숫자가 너무 작아서 로그로 크기를 맞춘다.
        hu = cv2.HuMoments(cv2.moments(contour)).flatten()
        return [
            float(-np.sign(value) * np.log10(abs(value) + 1e-12))
            for value in hu[:3]
        ]
