"""Actual mode-specific process logging, without network or model acquisition."""
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time


def records(path):
    values = [json.loads(line) for line in path.read_text().splitlines()]
    assert values and all(v["schema"] == "zerogravity.operational.v1" for v in values)
    assert [v["sequence"] for v in values] == list(range(len(values)))
    assert all(v["event"]["timestamp_unix_ms"] > 0 for v in values)
    return [v["event"] for v in values]


def run(binary, model=None):
    with tempfile.TemporaryDirectory(prefix="zg-operational-") as tmp:
        root = Path(tmp)
        env = dict(os.environ, HOME=tmp, XDG_CONFIG_HOME=str(root/"config"),
                   XDG_CACHE_HOME=str(root/"cache"), XDG_STATE_HOME=str(root/"state"),
                   ZEROGRAVITY_API_KEY="fixture-log-auth")
        env.pop("ZEROGRAVITY_LOG_FILE", None)
        env.pop("ZEROGRAVITY_LOG_FORMAT", None)
        env.pop("ZEROGRAVITY_LOG_LEVEL", None)
        # Ordinary JSON remains parseable and creates no log/state history.
        one = subprocess.run([binary, "--format", "json", "model", "list"],
                             env=env, cwd=root, capture_output=True, check=True)
        json.loads(one.stdout)
        assert not (root/"state").exists()
        # A real missing-source failure is timestamped on service stderr.
        args = [binary, "--offline", "serve", "--model", "source=sha256:"+"a"*64,
                "--backend", "cpu", "--listen", "127.0.0.1:0",
                "--max-context", "128", "--max-output", "8"]
        refused = subprocess.run(args, env=env, cwd=root, capture_output=True, timeout=30)
        assert refused.returncode != 0 and not refused.stdout
        assert re.search(rb"\d\d:\d\d:\d\d\.\d{3}Z ERROR", refused.stderr)
        assert b"code=" in refused.stderr and b"cause:" in refused.stderr
        assert b"fixture-log-auth" not in refused.stderr
        assert not (root/"state").exists()
        # Explicit file overrides stderr without creating an implicit session.
        path = root/"explicit.jsonl"
        refused = subprocess.run(args, env=dict(env, ZEROGRAVITY_LOG_FILE=str(path)),
                                 cwd=root, capture_output=True, timeout=30)
        assert refused.returncode != 0 and not refused.stdout
        assert path.stat().st_mode & 0o777 == 0o600
        events = records(path)
        assert any(e["level"] == "ERROR" and e["details"]["code"] and e["causes"] for e in events)
        assert not (root/"state").exists()
        errors = root/"errors.jsonl"
        refused = subprocess.run(args, env=dict(env, ZEROGRAVITY_LOG_FILE=str(errors), ZEROGRAVITY_LOG_LEVEL="error"),
                                 cwd=root, capture_output=True, timeout=30)
        assert refused.returncode != 0
        assert all(e["level"] == "ERROR" for e in records(errors))
        # Failure to open the requested file must be explicit and preserve
        # the real service error on the safe fallback stderr.
        refused = subprocess.run(args, env=dict(env, ZEROGRAVITY_LOG_FILE=str(root/"absent"/"log")),
                                 cwd=root, capture_output=True, timeout=30)
        assert refused.returncode != 0
        assert b"operational log unavailable" in refused.stderr and b"cause:" in refused.stderr
        print("Operational modes: one-off stdout purity, timestamped service stderr, explicit private file, visible fallback")
        if model is None:
            return
        # Optional approved actual GGUF route; no acquisition or fake service.
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1",0))
            port = reservation.getsockname()[1]
        args = [binary, "--offline", "serve", "--model", "source="+str(model),
                "--backend", "cpu", "--device", "cpu:0", "--listen", f"127.0.0.1:{port}",
                "--max-context", "128", "--max-output", "8", "--workers", "1",
                "--active-per-worker", "1"]
        path = root/"service.stderr"
        with path.open("wb") as stderr:
            process = subprocess.Popen(args, env=env, cwd=root, stdout=subprocess.PIPE, stderr=stderr)
            try:
                def request(method, path, body=None):
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    try:
                        connection.request(method, path, body=body,
                            headers={"Authorization":"Bearer fixture-log-auth","Content-Type":"application/json"})
                        response = connection.getresponse()
                        return response.status, response.read()
                    finally:
                        connection.close()
                deadline = time.monotonic()+180
                while time.monotonic()<deadline:
                    if process.poll() is not None:
                        raise RuntimeError("CPU service exited: "+path.read_text()[-4000:])
                    try:
                        status, data = request("GET","/v1/models")
                        if status == 200:
                            assert json.loads(data)["data"][0]["id"]=="source"
                            break
                    except OSError:
                        pass
                    time.sleep(.2)
                else:
                    raise TimeoutError("CPU service not ready")
                body=json.dumps({"model":"missing","prompt":"PRIVATE_LOG_FIXTURE_PROMPT","max_tokens":1})
                status,_=request("POST","/v1/completions",body)
                assert status==404,status
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    if "code=model_not_found" in path.read_text(): break
                    time.sleep(.1)
                else: raise AssertionError("HTTP error did not reach live stderr")
                process.send_signal(signal.SIGTERM)
                assert process.wait(timeout=60)==0,path.read_text()[-4000:]
                assert not process.stdout.read()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
        text=path.read_text()
        assert "shutdown complete" in text and "PRIVATE_LOG_FIXTURE_PROMPT" not in text
        assert "fixture-log-auth" not in text and not (root/"state").exists()
        print("Operational service: actual CPU listener, live request error without UI, joined retirement and no request body")


if __name__=="__main__":
    run(str(Path(sys.argv[1]).resolve()),Path(sys.argv[2]).resolve() if len(sys.argv)>2 else None)
