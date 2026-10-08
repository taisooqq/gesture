"""영상 프레임과 길이를 출력합니다.

src 폴더에서:
  python video_info.py
"""

from pathlib import Path

import cv2

from gesture.recorder import open_video

# 여기를 바꿔서 확인합니다.
FOLDER = Path(__file__).resolve().parents[1] / "data" / "videos" / "back"
NAME = "back_0b33bc.avi"


def main():
    path = FOLDER / NAME
    if not path.is_file():
        raise SystemExit(f"파일이 없습니다. {path}")

    cap = open_video(path)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다. {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        frames += 1
    cap.release()

    duration = frames / fps if fps > 0 else 0.0
    print(f"파일: {path}")
    print(f"프레임: {frames}")
    print(f"fps: {fps:.2f}")
    print(f"크기: {width} x {height}")
    print(f"길이: {duration:.2f}초")
    print("학습: 포함" if frames >= 1 else "학습: 프레임이 없어 빠짐")


if __name__ == "__main__":
    main()
