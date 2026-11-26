"""Real embedded control listener, security boundary and foreground retirement."""
import contextlib
import http.client
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit


class Client:
    def __init__(self, private_url):
        parsed = urlsplit(private_url)
        assert parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
        assert parsed.path == "/ui/" and not parsed.query
        self.address = parsed.netloc
        self.origin = f"http://{self.address}"
        self.port = parsed.port
        self.key = parsed.fragment.removeprefix("control=")
        assert len(self.key) == 64 and all(c in "0123456789abcdef" for c in self.key)

    def connection(self):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=8)

    def request(self, path, *, auth=True, headers=None, method="GET", body=None):
        fields = {"Authorization": "Bearer " + self.key} if auth else {}
        fields.update(headers or {})
        with contextlib.closing(self.connection()) as connection:
            connection.request(method, path, body=body, headers=fields)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()

    def snapshot(self):
        status, _, body = self.request("/zg/control/snapshot")
        assert status == 200
        assert self.key.encode() not in body
        return json.loads(body)

    def post(self, value, *, status=202, auth=True, headers=None):
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        fields = {"Origin": self.origin, "Content-Type": "application/json"}
        fields.update(headers or {})
        actual, _, data = self.request("/zg/control/actions", auth=auth, headers=fields, method="POST", body=body)
        assert actual == status, f"control action HTTP {actual}, expected {status}"
        return json.loads(data) if status == 202 else None

    def result(self, identity, *, state="completed", timeout=90):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, _, data = self.request("/zg/control/actions/" + identity)
            assert status == 200
            value = json.loads(data)
            assert value["id"] == identity
            if value["state"] != "running":
                assert value["state"] == state, value["message"]
                return value
            time.sleep(0.1)
        raise AssertionError("control action did not reach a terminal outcome")

    def action(self, value, *, state="completed"):
        return self.result(self.post(value)["id"], state=state)

    def observed(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.snapshot()
            if predicate(value):
                return value
            time.sleep(0.1)
        raise AssertionError("control snapshot did not reach expected owner state")

    @contextlib.contextmanager
    def events(self):
        with contextlib.closing(self.connection()) as connection:
            connection.request("GET", "/zg/events", headers={"Authorization": "Bearer " + self.key})
            response = connection.getresponse()
            assert response.status == 200
            assert response.getheader("Content-Type") == "text/event-stream"
            assert response.readline() == b"event: refresh\n"
            assert response.readline() == b"data: {}\n"
            assert response.readline() == b"\n"
            yield response

    def closed(self):
        with socket.socket() as connection:
            connection.settimeout(0.2)
            assert connection.connect_ex(("127.0.0.1", self.port)) != 0


def smoke(binary):
    with tempfile.TemporaryDirectory(prefix="zg-web-") as tmp:
        root = Path(tmp)
        profiles = root / "config/abird/zerogravity/profiles"
        profiles.mkdir(parents=True)
        profile = profiles / "desk.yaml"
        original = "schema: zerogravity.pipeline.v2\nrun:\n  task:\n    prompt: PRIVATE_PROMPT_NOT_IN_WIRE\n"
        profile.write_text(original)
        env = dict(os.environ, HOME=tmp, XDG_CONFIG_HOME=str(root / "config"),
                   XDG_CACHE_HOME=str(root / "cache"), ZEROGRAVITY_API_KEY="fixture-inference-only")
        invalid = subprocess.run([binary, "web", "--listen", "0.0.0.0:0"], env=env,
                                 cwd=root, capture_output=True)
        assert invalid.returncode != 0 and b"loopback" in invalid.stderr
        process = subprocess.Popen([binary, "--offline", "--profile", "desk", "web", "--listen", "127.0.0.1:0"],
                                   env=env, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert select.select([process.stdout], [], [], 10)[0], "control listener did not report its endpoint"
            line = process.stdout.readline().decode().strip()
            assert line.startswith("Web UI: "), "control listener failed to start"
            client = Client(line.removeprefix("Web UI: "))
            status, headers, html = client.request("/ui/", auth=False)
            assert status == 200 and headers["Content-Type"].startswith("text/html")
            assert headers["Cache-Control"] == "no-store"
            assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
            assert b"/ui/app.js" in html and b"/ui/style.css" in html
            for path, media in [("/ui/app.js", "text/javascript"), ("/ui/style.css", "text/css")]:
                status, headers, body = client.request(path, auth=False)
                assert status == 200 and headers["Content-Type"].startswith(media)
                assert client.key.encode() not in body
            status, _, license_text = client.request("/ui/licenses.txt", auth=False)
            assert status == 200 and b"MIT License" in license_text
            for path in ["/zg/control/snapshot", "/zg/events"]:
                assert client.request(path, auth=False)[0] == 401
                assert client.request(path, headers={"Authorization": "Bearer fixture-inference-only"})[0] == 401
                assert client.request(path, headers={"Host": "rebinding.invalid"})[0] == 400
                assert client.request(path, headers={"Origin": "http://127.0.0.1:1"})[0] == 403
                assert client.request(path, headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
            assert client.request("/zg/control/snapshot", headers={"Origin": client.origin})[0] == 200
            assert client.request("/zg/control/snapshot?control=" + client.key)[0] == 404
            assert client.request("/zg/control/snapshot", method="POST")[0] == 405
            assert client.request("/v1/models")[0] == 404
            assert client.request("/ui/../Cargo.toml")[0] == 404
            # Duplicate security headers and a nonzero request body refused.
            with socket.create_connection(("127.0.0.1", client.port), timeout=5) as sock:
                request = f"GET /zg/control/snapshot HTTP/1.1\r\nHost: {client.address}\r\nHost: {client.address}\r\n\r\n"
                sock.sendall(request.encode())
                # TCP does not preserve write boundaries. Require the full
                # refusal status line rather than a single recv's fragment.
                with sock.makefile("rb") as response:
                    assert response.readline(1024) == b"HTTP/1.1 400 Bad Request\r\n"
            assert client.request("/zg/control/snapshot", headers={"Content-Length": "1"})[0] == 400
            value = client.snapshot()
            assert value["schema"] == "zerogravity.control.v1"
            assert value["configuration"] == {"profile": "desk", "offline": True, "chat": True}
            assert value["runtime"]["status"]["phase"] == "empty"
            assert value["runtime"]["models"] == []
            assert "PRIVATE_PROMPT_NOT_IN_WIRE" not in json.dumps(value)
            assert "fixture-inference-only" not in json.dumps(value)
            assert value["capabilities"]["actions"] is True
            assert value["action"] is None and value["web_action"] is None
            # No untrusted body effects before the independent credential,
            # exact same-origin and bounded typed grammar checks.
            load = {"kind": "load", "source": ""}
            client.post(load, auth=False, status=401)
            client.post(load, headers={"Authorization": "Bearer fixture-inference-only"}, status=401)
            client.post(load, headers={"Origin": "http://127.0.0.1:1"}, status=403)
            client.post(load, headers={"Sec-Fetch-Site": "cross-site"}, status=403)
            assert client.request("/zg/control/actions", method="POST",
                                  headers={"Content-Type": "application/json"},
                                  body=json.dumps(load))[0] == 403
            client.post(load, headers={"Content-Type": "text/plain"}, status=400)
            client.post({"kind": "load", "source": "/tmp/model.gguf"}, status=400)
            client.post({"kind": "load", "source": "", "api_key": "private"}, status=400)
            client.post(b'{"kind":"load","source":"","source":"other"}', status=400)
            client.post({"kind": "unload", "alias": "model", "registration_id": "1", "confirm": False}, status=400)
            assert client.request("/zg/control/actions", method="POST",
                                  headers={"Origin": client.origin, "Content-Type": "application/json",
                                           "Content-Length": str(256 * 1024 + 1)})[0] == 413
            assert client.snapshot()["web_action"] is None
            # Actual original stale-registration refusal, not an invented
            # browser-side success. Only the exact action route retains a
            # bounded response; snapshots/events do not contain prompts.
            failed = client.action({"kind": "invoke", "alias": "missing", "registration_id": "1",
                                    "prompt": "PRIVATE_WEB_PROMPT_NOT_IN_WIRE", "chat": False}, state="failed")
            identity = failed["id"]
            assert client.request("/zg/control/actions/" + identity, auth=False)[0] == 401
            assert client.request("/zg/control/actions/0001")[0] == 404
            client.post({"kind": "cancel", "id": identity}, status=409)
            assert "PRIVATE_WEB_PROMPT_NOT_IN_WIRE" not in json.dumps(client.snapshot())
            assert client.snapshot()["runtime"]["status"]["phase"] == "empty"
            # Four SSE slots leave capacity for snapshots and refuse excess
            # streams; process retirement joins every stream and slow reader.
            with contextlib.ExitStack() as active:
                streams = [active.enter_context(client.events()) for _ in range(4)]
                assert client.request("/zg/events")[0] == 503
                assert client.snapshot()["runtime"]["status"]["phase"] == "empty"
                slow = active.enter_context(socket.create_connection(("127.0.0.1", client.port), timeout=5))
                slow.sendall(b"GET /zg/control/snapshot HTTP/1.1\r\n")
                process.send_signal(signal.SIGTERM)
                assert process.wait(timeout=8) == 0
                assert all(response.read() == b"" for response in streams)
            client.closed()
            assert profile.read_text() == original
            assert not (root / "cache").exists()
            print("Web control HTTP: assets, independent bearer and same-origin typed POST, bounded bodies/stale refusal, owner snapshot/SSE and joined retirement")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
            process.stderr.close()


if __name__ == "__main__":
    smoke(str(Path(sys.argv[1]).resolve()))
