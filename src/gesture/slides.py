"""슬라이드 창. 파이썬 표준 GUI인 tkinter로 카메라 창과 따로 띄웁니다."""

import tkinter as tk

from gesture.config import settings as default_settings


SLIDES = (
    ("비접촉 슬라이드", "주먹 = 다음, 보자기 = 이전"),
    ("수집", "클래스당 사람별 50장 이상, 중앙 박스만 저장"),
    ("특징", "피부색 분할 후 면적비, solidity, 결함 수, Hu"),
    ("실시간", "확신도가 낮거나 잠깐 흔들리면 명령을 보내지 않음"),
)


class SlideWindow:
    """제스처 명령을 받아 슬라이드를 넘기는 별도 창."""

    def __init__(self, settings=None):
        self.settings = settings or default_settings
        self.index = 0
        self._closed = False
        self.root = None
        self.status = None
        self.title = None
        self.body = None
        self.page = None

    @property
    def closed(self):
        return self._closed

    def start(self):
        root = tk.Tk()
        root.title("제스처 슬라이드")
        root.geometry("960x640")
        root.minsize(640, 420)
        root.configure(bg="#111111")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root = root

        header = tk.Frame(root, bg="#111111")
        header.pack(fill="x", padx=28, pady=(16, 0))
        self.status = tk.Label(
            header,
            text="제스처: -",
            bg="#111111",
            fg="#9ad7a8",
            font=("Helvetica", 16),
        )
        self.status.pack(side="left")

        buttons = tk.Frame(header, bg="#111111")
        buttons.pack(side="right")
        self._button(buttons, "이전", lambda: self.go(-1)).pack(side="left", padx=(0, 8))
        self._button(buttons, "다음", lambda: self.go(1)).pack(side="left")

        main = tk.Frame(root, bg="#111111")
        main.pack(fill="both", expand=True)
        self.title = tk.Label(
            main, text="", bg="#111111", fg="#f4f4f4", font=("Helvetica", 40)
        )
        self.title.pack(pady=(48, 16))
        self.body = tk.Label(
            main,
            text="",
            bg="#111111",
            fg="#cccccc",
            font=("Helvetica", 20),
            wraplength=760,
        )
        self.body.pack()

        self.page = tk.Label(
            root, text="", bg="#111111", fg="#888888", font=("Helvetica", 14)
        )
        self.page.pack(anchor="w", padx=28, pady=(0, 20))

        root.bind("<Left>", lambda _event: self.go(-1))
        root.bind("<Right>", lambda _event: self.go(1))
        self._render()
        root.update()
        root.lift()
        root.attributes("-topmost", True)
        root.update()
        root.attributes("-topmost", False)
        root.focus_force()
        print("슬라이드 창을 열었습니다.", flush=True)
        return root

    def _button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg="#222222",
            fg="#eeeeee",
            activebackground="#333333",
            activeforeground="#ffffff",
            relief="flat",
            padx=12,
            pady=6,
            font=("Helvetica", 13),
        )

    def go(self, delta):
        if self._closed:
            return
        self.index = (self.index + delta) % len(SLIDES)
        self._render()

    def _render(self):
        title, body = SLIDES[self.index]
        self.title.configure(text=title)
        self.body.configure(text=body)
        self.page.configure(text=f"{self.index + 1} / {len(SLIDES)}")

    def publish(self, gesture, conf, fps, action=None):
        if self._closed or self.root is None:
            return
        name = self.settings.gesture_ko.get(gesture, gesture)
        percent = round((conf or 0) * 100)
        self.status.configure(text=f"제스처: {name}  {percent}%  {fps or 0:.0f} fps")
        if action == "next":
            self.go(1)
        elif action == "prev":
            self.go(-1)
        self.pump()

    def pump(self):
        if self._closed or self.root is None:
            return
        try:
            self.root.update()
        except tk.TclError:
            self._closed = True

    def wait(self):
        print("슬라이드 창만 실행 중. 창을 닫으면 종료합니다.")
        if self._closed or self.root is None:
            return
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.shutdown()

    def _on_close(self):
        self.shutdown()

    def shutdown(self):
        root = self.root
        self.root = None
        self._closed = True
        if root is None:
            return
        try:
            root.destroy()
        except tk.TclError:
            pass
