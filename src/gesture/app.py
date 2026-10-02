"""수집, 학습, 데모를 한 진입점에서 고릅니다."""

import argparse

from gesture.collector import DataCollector
from gesture.live import LiveDemo
from gesture.trainer import GestureTrainer


class App:
    """python main.py collect | train | demo"""

    def run(self, argv=None):
        parser = argparse.ArgumentParser(description="손 제스처로 슬라이드를 조작합니다.")
        commands = parser.add_subparsers(dest="command", required=True)

        collect = commands.add_parser("collect", help="웹캠으로 손 이미지 저장")
        collect.add_argument("--person", default="p1", help="촬영자 ID")
        collect.add_argument("--camera", type=int, default=0)

        commands.add_parser("train", help="저장한 이미지로 모델 학습")

        demo = commands.add_parser("demo", help="실시간 인식으로 슬라이드 조작")
        demo.add_argument("--camera", type=int, default=0)
        demo.add_argument("--web-only", action="store_true", help="카메라 없이 슬라이드만")

        args = parser.parse_args(argv)
        if args.command == "collect":
            DataCollector(person=args.person, camera=args.camera).run()
        elif args.command == "train":
            GestureTrainer().run()
        else:
            LiveDemo(camera=args.camera, web_only=args.web_only).run()


def main():
    App().run()
