"""슬라이드 페이지와 현재 제스처 상태를 브라우저에 넘깁니다."""

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gesture.config import settings as default_settings


class SlideServer:
    """로컬 페이지를 띄우고, 확정된 명령만 /api/state 로 전달합니다."""

    def __init__(self, settings=None):
        self.settings = settings or default_settings
        self.state = {
            "gesture": "none",
            "action": "none",
            "conf": 0.0,
            "seq": 0,
            "fps": 0.0,
        }
        self._lock = threading.Lock()
        self._httpd = None

    def start(self, open_browser=True):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def _send(self, code, body, content_type):
                data = body if isinstance(body, bytes) else body.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/api/state":
                    with owner._lock:
                        payload = json.dumps(owner.state)
                    self._send(200, payload, "application/json")
                    return
                page = owner.settings.web_dir / "index.html"
                if path == "/" and page.exists():
                    self._send(200, page.read_bytes(), "text/html; charset=utf-8")
                    return
                self._send(404, "not found", "text/plain")

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.settings.web_port), Handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{self.settings.web_port}"
        print("슬라이드:", url)
        if open_browser:
            webbrowser.open(url)
        return self._httpd

    def publish(self, gesture, conf, fps, action=None):
        with self._lock:
            self.state["gesture"] = gesture
            self.state["conf"] = round(conf, 3)
            self.state["fps"] = round(fps, 1)
            if action:
                self.state["action"] = action
                self.state["seq"] += 1
            else:
                self.state["action"] = "none"

    def shutdown(self):
        if self._httpd is not None:
            self._httpd.shutdown()
