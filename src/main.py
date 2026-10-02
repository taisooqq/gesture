"""프로젝트 실행 파일.

src 폴더에서:
  python main.py collect --person p1
  python main.py train
  python main.py demo
"""

from gesture.app import App


if __name__ == "__main__":
    App().run()
