"""실행하면 인식·등록 창을 엽니다."""

import argparse


class App:
    """python main.py"""

    def run(self, argv=None):
        parser = argparse.ArgumentParser(description="손 제스처 인식과 영상 등록 창을 엽니다.")
        parser.add_argument("--camera", type=int, default=0, help="카메라 번호")
        args = parser.parse_args(argv)
        from gesture.ui import GestureWindow

        GestureWindow(camera=args.camera).run()


def main():
    App().run()
