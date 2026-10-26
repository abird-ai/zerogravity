#!/usr/bin/env python3
"""Actual TERM/INT controls for the model-free Rust serving fixture."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import time


def exited(process):
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)


def members(group):
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
            if int(fields[2]) == group:
                found.append((int(path.name), fields[0], int(fields[3])))
        except FileNotFoundError:
            continue
        except PermissionError:
            continue
    return found


def events(path):
    data = path.read_bytes()
    if len(data) > 128 * 1024:
        raise AssertionError("serving fixture output exceeds its control bound")
    result = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                row = json.loads(line[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("event") in {"READY", "CLOSED"}:
                result.append(row)
    return result


def health(address):
    host, port = address.rsplit(":", 1)
    connection = http.client.HTTPConnection(host, int(port), timeout=2)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200, response.status
        document = json.loads(response.read(64 * 1024))
        assert document["status"] == "ok", document
        return host, int(port)
    finally:
        connection.close()


def run_case(binary, child_test, namespace, sig, directory):
    directory.mkdir(mode=0o700)
    output, errors, information = [directory / name for name in ("stdout.log", "stderr.log", "namespace.json")]
    command = [str(binary), "--exact", child_test, "--ignored", "--nocapture", "--test-threads=1", "--format=terse"]
    environment = {**os.environ, "ZG_SERVE_SHUTDOWN_FIXTURE": "1"}
    sockets, process, target_fd, info_fd = [], None, None, None
    result = {"namespace_pid1": namespace, "signal": sig.name, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    failure = None
    try:
        if namespace:
            wrapper = shutil.which("bwrap")
            assert wrapper, "PID1 serving controls require bubblewrap"
            info_fd = os.open(information, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            command = [wrapper, "--die-with-parent", "--unshare-pid", "--as-pid-1", "--info-fd", str(info_fd),
                       "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev", "--", *command]
        with output.open("xb") as stdout, errors.open("xb") as stderr:
            os.chmod(output, 0o600)
            os.chmod(errors, 0o600)
            process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, start_new_session=True,
                                       pass_fds=() if info_fd is None else (info_fd,))
        if info_fd is not None:
            os.close(info_fd)
            info_fd = None
        deadline = time.monotonic() + 15
        ready = None
        while time.monotonic() < deadline:
            observed = events(output)
            ready = next((row for row in observed if row["event"] == "READY"), None)
            if ready is not None:
                break
            assert exited(process) is None, "serving fixture exited before readiness"
            time.sleep(.02)
        assert ready is not None, "serving fixture did not become ready"
        target = json.loads(information.read_text())["child-pid"] if namespace else process.pid
        target_fd = os.pidfd_open(target)
        assert ready["pid"] == (1 if namespace else process.pid), ready
        address = health(ready["address"])
        # An idle socket, unfinished headers, and an unfinished declared body
        # must all retire; a later health response proves earlier admission.
        for payload in (b"", b"POST /v1/completions HTTP/1.1\r\nHost: fixture\r\n",
                        b"POST /v1/completions HTTP/1.1\r\nHost: fixture\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{"):
            client = socket.create_connection(address, timeout=2)
            sockets.append(client)
            if payload:
                client.sendall(payload)
        health(ready["address"])
        assert exited(process) is None, "serving fixture lost its process owner"
        started = time.monotonic()
        signal.pidfd_send_signal(target_fd, sig)
        while exited(process) is None and time.monotonic() - started < 4:
            time.sleep(.01)
        assert exited(process) is not None, "serving shutdown exceeded four seconds"
        closed = [row for row in events(output) if row["event"] == "CLOSED"]
        assert len(closed) == 1 and closed[0]["serving_host_bytes"] == 0, closed
        for client in sockets:
            client.settimeout(.5)
            # A bounded error response may precede EOF. Drain it, then require
            # closure rather than mistaking its first bytes for a live handler.
            remaining = 64 * 1024
            while remaining:
                try:
                    block = client.recv(min(4096, remaining))
                except ConnectionResetError:
                    break
                if not block:
                    break
                remaining -= len(block)
            else:
                raise AssertionError("shutdown socket output exceeded its bound")
        result.update(pid=ready["pid"], target_pid=target, shutdown_seconds=time.monotonic() - started,
                      joined=True, serving_host_bytes=closed[0]["serving_host_bytes"])
    except BaseException as error:
        failure = error
    finally:
        if info_fd is not None:
            os.close(info_fd)
        for client in sockets:
            client.close()
        if process is not None:
            live_before_cleanup = [row for row in members(process.pid) if row[1] != "Z"]
            result["live_members_before_cleanup"] = live_before_cleanup
            result["cleanup_signals"] = []
            if failure is None and live_before_cleanup:
                failure = AssertionError("serving fixture left live group members after CLOSED")
            # Hold the original child unreaped while signaling its group. The
            # pidfd also keeps namespace-init signaling immune to PID reuse.
            if failure is not None and target_fd is not None:
                try:
                    signal.pidfd_send_signal(target_fd, signal.SIGKILL)
                    result["cleanup_signals"].append("target:SIGKILL")
                except ProcessLookupError:
                    pass
            for sig_cleanup in (signal.SIGTERM, signal.SIGKILL):
                live = [row for row in members(process.pid) if row[1] != "Z"]
                if not live:
                    break
                try:
                    os.killpg(process.pid, sig_cleanup)
                    result["cleanup_signals"].append("group:" + sig_cleanup.name)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 1
                while any(row[1] != "Z" for row in members(process.pid)) and time.monotonic() < deadline:
                    time.sleep(.01)
            code = process.wait(timeout=2)
            result.update(returncode=code, group_absent=not members(process.pid))
            if failure is None and (code != 0 or not result["group_absent"]):
                failure = AssertionError("serving child did not close its original owner cleanly")
        if target_fd is not None:
            os.close(target_fd)
        result.update(stdout=str(output), stderr=str(errors), passed=failure is None)
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if failure is not None:
        raise AssertionError(f"{result}; stderr={errors.read_text(errors='replace')}") from failure
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-binary", type=Path, required=True)
    parser.add_argument("--child-test", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    # Retain actual process output for failed or successful cases under the
    # caller's test TMPDIR; the outer supervised run owns that directory.
    directory = Path(tempfile.mkdtemp(prefix="zg-serving-shutdown-"))
    results = []
    for namespace in (False, True):
        for sig in (signal.SIGTERM, signal.SIGINT):
            tag = ("pid1" if namespace else "process") + "-" + sig.name.lower()
            results.append(run_case(args.test_binary.resolve(strict=True), args.child_test, namespace, sig, directory / tag))
    print(json.dumps({"serving_shutdown_controls": results, "evidence": str(directory)}), flush=True)


if __name__ == "__main__":
    main()
