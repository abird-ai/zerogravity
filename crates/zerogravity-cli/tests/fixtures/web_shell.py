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

    def inventory(self, name):
        status, _, data = self.request("/zg/control/inventory/" + name)
        assert status == 200
        assert self.key.encode() not in data
        return json.loads(data)

    def config(self, name):
        status, _, data = self.request("/zg/control/config/" + name)
        assert status == 200
        assert self.key.encode() not in data
        return json.loads(data)

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
            try:
                yield response
            finally:
                response.close()

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
                   XDG_CACHE_HOME=str(root / "cache"), XDG_STATE_HOME=str(root / "state"),
                   ZEROGRAVITY_API_KEY="fixture-inference-only")
        env.pop("ZEROGRAVITY_LOG_FILE", None)
        env.pop("ZEROGRAVITY_LOG_FORMAT", None)
        env.pop("ZEROGRAVITY_LOG_LEVEL", None)
        log_root = root / "state/abird/zerogravity/logs"

        def records():
            files = sorted(log_root.glob("session-*/events-*.jsonl"))
            assert files and all(path.stat().st_mode & 0o777 == 0o600 for path in files)
            assert all(path.parent.stat().st_mode & 0o777 == 0o700 for path in files)
            values = [json.loads(line) for path in files for line in path.read_text().splitlines()]
            assert values and all(value["schema"] == "zerogravity.operational.v1" for value in values)
            assert [v["sequence"] for v in values] == list(range(len(values)))
            assert all(v["event"]["timestamp_unix_ms"] > 0 for v in values)
            assert all(secret not in json.dumps(values) for secret in [
                "PRIVATE_PROMPT_NOT_IN_WIRE", "PRIVATE_WEB_PROMPT_NOT_IN_WIRE",
                "PRIVATE_COMMENT_NOT_IN_WIRE", "fixture-inference-only",
                client.key,
            ])
            return [value["event"] for value in values]
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
            for path in ["/zg/control/snapshot", "/zg/events", "/zg/control/inventory/models", "/zg/control/inventory/devices", "/zg/control/inventory/profiles", "/zg/control/inventory/logs", "/zg/control/config/@active"]:
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
            assert value["configuration"] == {"profile": "desk", "offline": True, "chat": True, "theme": "auto"}
            assert value["runtime"]["status"]["phase"] == "empty"
            assert value["runtime"]["models"] == []
            assert "PRIVATE_PROMPT_NOT_IN_WIRE" not in json.dumps(value)
            assert "fixture-inference-only" not in json.dumps(value)
            assert value["capabilities"]["actions"] is True
            assert value["action"] is None and value["web_action"] is None
            models = client.inventory("models")
            assert models["cached"] == models["partial"] == []
            assert any(row["slug"] == "qwen3.5-0.8b" for row in models["catalog"])
            devices = client.inventory("devices")
            assert any(row["id"] == "cpu:0" for row in devices["devices"])
            assert devices["observed_unix_ms"].isdecimal()
            assert client.request("/zg/control/inventory/models", method="POST")[0] == 405
            digest = "a" * 64
            for invalid in [
                {"kind": "inspect", "digest": "a" * 12},
                {"kind": "verify", "digest": digest.upper()},
                {"kind": "alias", "digest": digest, "name": "../alias"},
                {"kind": "remove", "digest": digest, "confirm": False},
                {"kind": "check_fit", "digest": digest, "device": "/dev/dri/renderD128"},
                {"kind": "load_on_device", "digest": digest, "device": "gpu:0"},
                {"kind": "pull", "source": "https://example.invalid/model", "selection": "model.gguf", "confirm": True},
                {"kind": "pull", "source": "hf:fixture/model@main", "selection": "model.gguf", "confirm": True},
                {"kind": "download", "operation_id": 1, "command": "cancel"},
            ]:
                client.post(invalid, status=400)
            if value["capabilities"]["acquisition"]:
                source = "hf:fixture/model@" + "1" * 40
                client.post({"kind": "pull", "source": source, "selection": "model.gguf", "confirm": False}, status=400)
                client.post({"kind": "pull", "source": source, "selection": "../model.gguf", "confirm": True}, status=400)
                client.post({"kind": "pull", "source": source, "selection": "model.gguf", "confirm": True}, status=403)
                client.post({"kind": "download", "operation_id": "1", "command": "cancel"}, status=409)
                assert client.snapshot()["download"] is None
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
            live_failure = next(event for event in records()
                if event["level"] == "ERROR" and event["correlation"].startswith("action #" + identity + " ·"))
            assert live_failure["details"]["code"] and live_failure["stage"]
            assert live_failure["elapsed_ms"] >= 0
            if value["capabilities"]["cache"]:
                inspected = client.action({"kind": "inspect", "digest": digest}, state="failed")
                local_failure = next(event for event in records()
                    if event["level"] == "ERROR" and event["correlation"].startswith("action #" + inspected["id"] + " ·"))
                assert local_failure["causes"] and "sha256:" + digest in local_failure["correlation"]
                assert all("causes" not in row for row in client.inventory("logs")["entries"])
            if value["capabilities"]["acquisition"]:
                preview = client.action({"kind": "pull_preview", "source": source, "selection": "model.gguf"})
                assert source.removeprefix("hf:") in preview["message"]
                assert "model.gguf" in preview["message"]
                assert client.snapshot()["download"] is None
                quant = client.action({"kind": "pull_preview", "source": source, "selection": "quant:Q4_K_M"})
                assert "Q4_K_M" in quant["message"]
                recipe = next((row for row in models["catalog"] if row["recipe"]), None)
                if recipe:
                    preview = client.action({"kind": "pull_preview", "source": recipe["slug"], "selection": ""})
                    assert recipe["recipe"]["snapshot"] in preview["message"]
            # The real typed owner, never a browser-selected filesystem path.
            assert client.inventory("profiles") == {"profiles": ["default", "desk"], "active": "desk"}
            protected = client.config("@active")
            assert protected["raw"] is None and protected["active"]
            assert "PRIVATE_PROMPT_NOT_IN_WIRE" not in json.dumps(protected)
            assert str(root) not in json.dumps(protected)
            def setting(command, **fields):
                return {"kind": "settings", "action": {"command": command, **fields}}
            client.post(setting("create", name="../outside"), status=400)
            client.post(setting("set", snapshot="0" * 32, key="ui.color", value="never"), status=409)
            client.post(setting("set", snapshot=protected["snapshot"], key="run.task.prompt", value="secret"), status=403)
            client.post(setting("replace", snapshot=protected["snapshot"], yaml="schema: zerogravity.pipeline.v2"), status=403)
            client.action(setting("remove", snapshot=protected["snapshot"], confirm=True), state="failed")
            assert profile.read_text() == original
            with client.events() as stream:
                client.action(setting("create", name="webdesk"))
                for _ in range(24):
                    if stream.readline() == b"event: configuration\n":
                        break
                else:
                    raise AssertionError("no configuration invalidation")
            webdesk = profiles / "webdesk.yaml"
            first = client.config("webdesk")
            assert first["raw"] is not None and first["exists"] and not first["active"]
            # Out-of-process edit: the original ConfigFile must refuse stale bytes.
            outside = "schema: zerogravity.pipeline.v2\n# PRIVATE_COMMENT_NOT_IN_WIRE\nui: {color: always}\n"
            webdesk.write_text(outside)
            client.action(setting("set", snapshot=first["snapshot"], key="ui.color", value="never"), state="failed")
            assert webdesk.read_text() == outside
            current = client.config("webdesk")
            assert current["snapshot"] != first["snapshot"]
            assert "PRIVATE_COMMENT_NOT_IN_WIRE" not in json.dumps(current)
            client.post(setting("set", snapshot=first["snapshot"], key="ui.color", value="never"), status=409)
            client.post(setting("set", snapshot=current["snapshot"], key="run.device", value="[/dev/dri/renderD128]"), status=403)
            client.post(setting("replace", snapshot=current["snapshot"], yaml="schema: zerogravity.pipeline.v2\nrun: {task: {prompt: hidden}}"), status=403)
            client.action(setting("set", snapshot=current["snapshot"], key="ui.color", value="never"))
            assert "color: never" in webdesk.read_text()
            current = client.config("webdesk")
            client.action(setting("unset", snapshot=current["snapshot"], key="ui.color", confirm=True))
            current = client.config("webdesk")
            client.action(setting("replace", snapshot=current["snapshot"],
                                  yaml="schema: zerogravity.pipeline.v2\nrun: {backend: cpu, task: {raw: true}}"))
            current = client.config("webdesk")
            client.action(setting("validate", snapshot=current["snapshot"]))
            client.action(setting("activate", snapshot=current["snapshot"]))
            assert client.snapshot()["configuration"] == {"profile": "webdesk", "offline": True, "chat": False, "theme": "auto"}
            client.action(setting("remove", snapshot=current["snapshot"], confirm=True), state="failed")
            client.action(setting("copy", snapshot=current["snapshot"], name="webcopy"))
            copied = profiles / "webcopy.yaml"
            assert copied.read_bytes() == webdesk.read_bytes()
            copy = client.config("webcopy")
            client.post(setting("remove", snapshot=copy["snapshot"], confirm=False), status=400)
            client.action(setting("remove", snapshot=copy["snapshot"], confirm=True))
            assert not copied.exists()
            default = client.config("default")
            assert not default["exists"] and not default["removable"]
            client.action(setting("remove", snapshot=default["snapshot"], confirm=True), state="failed")
            desk = client.config("desk")
            client.action(setting("activate", snapshot=desk["snapshot"]))
            assert client.snapshot()["configuration"]["profile"] == "desk"
            logs = client.inventory("logs")
            assert logs["sink_available"] is True
            assert all(row["timestamp_unix_ms"] > 0 for row in logs["entries"])
            assert len(logs["entries"]) <= 128
            assert any(row["message"] == "Save config completed" for row in logs["entries"])
            assert any(row["message"] == "Save config returned error" for row in logs["entries"])
            assert all(secret not in json.dumps(logs) for secret in ["PRIVATE_PROMPT_NOT_IN_WIRE", "PRIVATE_WEB_PROMPT_NOT_IN_WIRE", "PRIVATE_COMMENT_NOT_IN_WIRE", "fixture-inference-only", str(root)])
            assert not (profiles.parent / "config.yaml").exists()
            print("Web settings HTTP: shared create/copy/activate/validate/set/unset/raw/remove, stale file/token and protected-field refusal, redaction, configuration SSE, bounded operational logs")
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
            assert any(event["message"] == "Web control host retired" for event in records())
            print("Operational logs: live private JSONL, typed terminal errors, safe browser projection and persisted joined shutdown")
            assert profile.read_text() == original
            assert not (root / "cache").exists()
            print("Web control HTTP: assets, independent bearer, bounded typed POST, cold model/device inventory, offline/stale refusal, owner snapshot/SSE and joined retirement")
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


def controlled_download(binary):
    # The ignored child executes the real protected transfer with the
    # existing deterministic transport fixture. There is no production
    # URL override and no external network or provider credentials.
    for mode in ("web", "web-cancel"):
        with tempfile.TemporaryDirectory(prefix="zg-web-download-") as tmp:
            root = Path(tmp)
            cache = root / "cache/abird/zerogravity/models"
            env = dict(os.environ, HOME=tmp, XDG_CONFIG_HOME=str(root / "config"),
                       XDG_CACHE_HOME=str(root / "cache"),
                       ZG_MONITOR_FIXTURE=str(cache), ZG_MONITOR_MODE=mode)
            process = subprocess.Popen(
                [binary, "--ignored", "--exact", "pull::monitor::tests::monitor_child", "--nocapture"],
                env=env, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    assert select.select([process.stdout], [], [], 10)[0], "owned child did not report Web URL"
                    line = process.stdout.readline().decode().strip()
                    assert line or process.poll() is None, "owned child terminated"
                    if "Web UI: " in line:
                        client = Client(line.split("Web UI: ", 1)[1])
                        break
                else:
                    raise AssertionError("owned child did not start its Web host")
                paused = client.observed(lambda row: row["download"]["status"] == "paused")["download"]
                identity = paused["id"]

                def command(name, *, status=202, operation=identity):
                    result = client.post({"kind": "download", "operation_id": operation, "command": name},
                                         status=status)
                    if status == 202:
                        assert result == {"operation_id": identity}
                    return result

                command("resume", status=409, operation=str(int(identity) + 1))
                command("resume")
                client.observed(lambda row: row["download"]["total_bytes"] > 0)
                command("pause")
                second_pause = client.observed(lambda row: row["download"]["status"] == "paused")["download"]
                assert second_pause["id"] == identity
                assert second_pause["total_bytes"] > 0
                assert not second_pause["terminal"]
                assert second_pause["checkpoint_bytes"] > 0
                # Checkpoint observation is not a grant to resume arbitrary
                # files. The live operation remains the only command target.
                assert client.inventory("models")["cached"] == []
                command("pause", status=409)
                with client.events() as events:
                    command("cancel" if mode == "web-cancel" else "resume")
                    terminal = client.observed(lambda row: row["download"]["terminal"], timeout=15)["download"]
                    assert terminal["id"] == identity
                    assert terminal["status"] == ("cancelled" if mode == "web-cancel" else "durable")
                    for _ in range(80):
                        if events.readline() == b"event: models\n":
                            break
                    else:
                        raise AssertionError("terminal transfer did not invalidate cold model inventory")
                models = client.inventory("models")
                if mode == "web-cancel":
                    assert models["cached"] == []
                    assert models["partial"] and models["partial"][0]["bytes"] > 0
                else:
                    assert len(models["cached"]) == 1
                    digest = terminal["snapshot"]
                    assert models["cached"][0]["digest"] == digest
                    assert models["cached"][0]["payload_verified"] is False
                    assert "verified" in client.action({"kind": "verify", "digest": digest})["message"]
                    refused = client.action({"kind": "remove", "digest": digest, "confirm": True}, state="failed")
                    assert "references" in refused["message"]
                    assert client.inventory("models")["cached"][0]["digest"] == digest
                command("cancel", status=409)
                assert client.snapshot()["action"] is None
                assert b"NOT_IN_OUTPUT" not in json.dumps(client.snapshot()).encode()
                process.stdin.write(b"done\n")
                process.stdin.flush()
                assert process.wait(timeout=10) == 0, process.stderr.read().decode(errors="replace")
                client.closed()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdin.close()
                process.stdout.close()
                process.stderr.close()
    print("Web controlled transfer: exact-operation pause/resume/cancel, checkpoint/terminal observations, stale refusal, owner invalidation and referenced cache protection")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--download-fixture":
        controlled_download(str(Path(sys.argv[1]).resolve()))
    else:
        smoke(str(Path(sys.argv[1]).resolve()))
