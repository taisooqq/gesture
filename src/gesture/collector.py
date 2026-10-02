"""웹캠으로 제스처 이미지를 클래스 폴더에 저장합니다."""

import cv2

from gesture.config import settings as default_settings
from gesture.hand import HandAnalyzer


class DataCollector:
    """키를 누를 때마다 중앙 ROI를 data/{클래스}/ 에 저장합니다."""

    def __init__(self, person="p1", camera=0, settings=None, analyzer=None):
        self.person = person
        self.camera = camera
        self.settings = settings or default_settings
        self.analyzer = analyzer or HandAnalyzer(self.settings)
        self.selected = 0

    def run(self):
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            raise SystemExit("웹캠을 열 수 없습니다.")

        counts = self._counts()
        print("촬영자:", self.person)
        print("키 1~5로 클래스, Space 저장, Q 종료")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if not self._step(cv2.flip(frame, 1), counts):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _step(self, frame, counts):
        gesture = self.settings.gestures[self.selected]
        roi, (x, y, w, h) = self.analyzer.crop(frame)
        view = frame.copy()
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 220, 0), 2)
        ko = self.settings.gesture_ko[gesture]
        label = f"{self.person}  {gesture} ({ko})  saved={counts[gesture]}"
        cv2.putText(view, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("collect", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if ord("1") <= key < ord("1") + len(self.settings.gestures):
            self.selected = key - ord("1")
        elif key == ord(" "):
            self._save(gesture, roi, counts)
        return True

    def _save(self, gesture, roi, counts):
        folder = self.settings.data_dir / gesture
        path = folder / f"{self.person}_{gesture}_{self._next_index(folder, gesture):04d}.png"
        if self._write_png(path, roi):
            counts[gesture] += 1
            print("saved", path)

    def _counts(self):
        counts = {}
        for gesture in self.settings.gestures:
            folder = self.settings.data_dir / gesture
            pattern = f"{self.person}_{gesture}_*.png"
            counts[gesture] = len(list(folder.glob(pattern))) if folder.exists() else 0
        return counts

    def _next_index(self, folder, gesture):
        nums = []
        if folder.exists():
            for path in folder.glob(f"{self.person}_{gesture}_*.png"):
                try:
                    nums.append(int(path.stem.rsplit("_", 1)[-1]))
                except ValueError:
                    continue
        return (max(nums) + 1) if nums else 1

    def _write_png(self, path, image):
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
        return True
