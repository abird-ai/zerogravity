"""Real PTY checks; optional cached-model lifecycle uses the caller's fixture."""
import fcntl
import os
from pathlib import Path
import pty
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import termios
import time
import urllib.request
import json

binary = str(Path(sys.argv[1]).resolve())


def environment(root):
    env = dict(os.environ, HOME=str(root), XDG_CONFIG_HOME=str(root / "config"),
               XDG_CACHE_HOME=str(root / "cache"), TERM="xterm-256color", NO_COLOR="1")
    env.pop("ZEROGRAVITY_API_KEY", None)
    return env


class Terminal:
    def __init__(self, root, args, width=100, height=28):
        self.master, self.slave = pty.openpty()
        self.height = height
        self.resize(width)
        self.initial = termios.tcgetattr(self.slave)
        self.proc = subprocess.Popen([binary, *args], cwd=root, env=environment(root),
                                     stdin=self.slave, stdout=self.slave,
                                     stderr=self.slave, start_new_session=True)
        self.pending = bytearray()
        self.transcript = bytearray()

    def resize(self, width):
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ,
                    struct.pack("HHHH", self.height, width, 0, 0))

    def until(self, needle, seconds=10):
        deadline = time.monotonic() + seconds
        while needle not in self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"missing {needle!r}: {self.transcript[-4000:]!r}")
            if select.select([self.master], [], [], min(remaining, 0.1))[0]:
                data = os.read(self.master, 65536)
                self.pending.extend(data)
                self.transcript.extend(data)
        del self.pending[:self.pending.index(needle) + len(needle)]

    def send(self, value, expected, seconds=10):
        self.pending.clear()
        while select.select([self.master], [], [], 0)[0]:
            self.transcript.extend(os.read(self.master, 65536))
        os.write(self.master, value)
        self.until(expected, seconds)

    def check_started(self):
        self.until(b"\x1b[?1049h")
        self.until(b"owners")
        assert termios.tcgetattr(self.slave) != self.initial

    def finish(self, *, interactive=True):
        if interactive:
            self.until(b"\x1b[?1049l")
        assert self.proc.wait(timeout=60) == 0
        assert termios.tcgetattr(self.slave) == self.initial

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        os.close(self.slave)
        os.close(self.master)


def shell(root):
    def exercise(args, actions, *, interactive=True, terminate=False):
        with Terminal(root, args) as terminal:
            if interactive:
                terminal.check_started()
                for command, expected in actions:
                    if isinstance(command, int):
                        terminal.resize(command)
                        os.kill(terminal.proc.pid, signal.SIGWINCH)
                        terminal.until(expected)
                    else:
                        terminal.send(command, expected)
                if terminate:
                    terminal.proc.send_signal(signal.SIGTERM)
                else:
                    os.write(terminal.master, b"q")
            else:
                terminal.until(b"USAGE:")
                assert b"\x1b[?1049h" not in terminal.transcript
            terminal.finish(interactive=interactive)

    exercise([], [(b"8", b"defaults"), (b":q", b"matching"),
                  (b"\x7flog\r", b"received."),
                  (60, b"ZeroGravity"), (b"\t", b"Navigation")])
    exercise([], [], terminate=True)
    exercise(["--offline"], [(b"4", b"Cached"), (b"m", b"qwen3.5-plus"),
                              (b"d", b"prohibits"), (b"5", b"Download")])
    exercise(["--help"], [], interactive=False)
    exercise(["--interactive", "never"], [], interactive=False)
    plain = subprocess.run([binary], cwd=root, env=environment(root),
                           stdin=subprocess.DEVNULL, capture_output=True, check=True)
    assert b"USAGE:" in plain.stdout and b"\x1b" not in plain.stdout
    assert not (root / "config").exists() and not (root / "cache").exists()
    profile = root / "config/abird/zerogravity/profiles/plain.yaml"
    profile.parent.mkdir(parents=True)
    document = "schema: zerogravity.pipeline.v2\nui:\n  interactive: never\n"
    profile.write_text(document)
    exercise(["--profile", "plain"], [], interactive=False)
    exercise(["--profile", "plain", "--interactive", "always"], [])
    assert profile.read_text() == document
    assert not (profile.parent.parent / "config.yaml").exists()
    assert not (root / "cache").exists()
    print("TUI shell PTY: navigation, palette, resize, quit/SIGTERM restoration, plain fallback; no cache/config effects")


def lifecycle(root, expected):
    # Caller prepared the immutable approved GGUF cache and profile. Use one
    # process for quick run, reused invocation, listener, retirement/reload.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{reservation.getsockname()[1]}"
    base = f"http://{address}/v1"
    prompt = "Continue: 1, 2, 3,"

    def http(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(base + path, data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    def closed():
        with socket.socket() as connection:
            connection.settimeout(0.2)
            host, port = address.split(":")
            assert connection.connect_ex((host, int(port))) != 0

    with Terminal(root, ["--profile", "local"], width=150, height=40) as terminal:
        terminal.check_started()
        terminal.send(b"6", b"cpu:0", 30)
        # Original inventory ordering ends in host memory. Use the original
        # read-only CLI observation to select that exact physical row.
        observation = subprocess.run([binary, "device", "list", "--format=json"], cwd=root,
                                     env=environment(root), capture_output=True, check=True)
        devices = json.loads(observation.stdout)["devices"]
        cpu_index = next(i for i, device in enumerate(devices) if device["id"] == "cpu:0")
        if cpu_index:
            os.write(terminal.master, b"j" * cpu_index)
        terminal.send(b"e", b"MODEL")
        terminal.send(b"\r", b"accepted", 60)
        terminal.send(b"4", b"fixture/serve")
        terminal.send(b"\r", b"checked;")
        terminal.send(b"v", b"verified", 60)
        terminal.send(b"a", b"replacement.")
        terminal.send(b"tui-fixture\r", b"created", 30)
        terminal.send(b"x", b"confirm")
        os.write(terminal.master, b"n")
        terminal.send(b"p", b"Prompt")
        # Original profile raw=true. TUI completion must be actual output.
        terminal.send(b"\x1b[200~" + prompt.encode() + b"\x1b[201~\r", b"length", 90)
        assert b"Response" in terminal.transcript
        terminal.send(b"2", b"registration")
        terminal.send(b"\r", b"Prompt")
        terminal.send(b"\x1b[200~" + prompt.encode() + b"\x1b[201~\r", b"length", 30)
        terminal.send(b"3", b"OpenAI-compatible")
        terminal.send(b"s", b"IP:PORT")
        terminal.send(address.encode() + b"\r", b"Listening:", 30)
        models = http("/models")
        assert len(models["data"]) == 1, models
        loaded_alias = models["data"][0]["id"]
        assert len(loaded_alias) == 64 and all(c in "0123456789abcdef" for c in loaded_alias), models
        result = http("/completions", {"model": loaded_alias, "prompt": prompt,
                                       "max_tokens": 2, "stream": False})
        assert result["choices"][0]["text"] == expected, result
        assert result["usage"]["completion_tokens"] > 0
        # Navigation and a declined quit leave the same listener alive.
        terminal.send(b"1", b"Quick")
        terminal.send(b"q", b"confirm")
        os.write(terminal.master, b"n")
        assert http("/models") == models
        terminal.send(b"3", b"OpenAI-compatible")
        terminal.send(b"t", b"confirm")
        terminal.send(b"y", b"retained")
        closed()
        terminal.send(b"2", b"registered")
        terminal.send(b"u", b"confirm")
        terminal.send(b"y", b"retired", 60)
        # The last retirement makes the same runtime available for a new load.
        terminal.send(b"l", b"MODEL")
        terminal.send(b"\r", b"loaded", 90)
        terminal.send(b"3", b"OpenAI-compatible")
        terminal.send(b"s", b"IP:PORT")
        terminal.send(address.encode() + b"\r", b"Listening:", 30)
        assert [model["id"] for model in http("/models")["data"]] == ["configured"]
        terminal.send(b"q", b"confirm")
        os.write(terminal.master, b"y")
        terminal.finish()
        closed()
    alias = subprocess.run([binary, "model", "inspect", "by-name:tui-fixture", "--format=json"],
                           cwd=root, env=environment(root), capture_output=True, check=True)
    assert json.loads(alias.stdout)["snapshot"] == loaded_alias
    with Terminal(root, ["--profile", "local"], width=150, height=40) as terminal:
        terminal.check_started()
        terminal.send(b"4", b"fixture/serve")
        terminal.send(b"X", b"preview;", 30)
        terminal.send(b"x", b"confirm")
        terminal.send(b"y", b"durable", 60)
        os.write(terminal.master, b"q")
        terminal.finish()
    listing = subprocess.run([binary, "model", "list", "--format=json"], cwd=root,
                             env=environment(root), capture_output=True, check=True)
    assert json.loads(listing.stdout) == []
    print("TUI cached lifecycle: inspect/verify, alias, declined removal, exact-digest run, repeated invoke, HTTP equality, stop/unload/reload/quit, preview and durable removal")


if len(sys.argv) == 2:
    with tempfile.TemporaryDirectory(prefix="zg-tui-") as tmp:
        shell(Path(tmp))
elif sys.argv[2] == "--download-fixture":
    for mode in ("tui", "tui-cancel"):
        with tempfile.TemporaryDirectory(prefix="zg-tui-download-") as tmp:
            root = Path(tmp)
            os.environ["ZG_MONITOR_FIXTURE"] = str(root / "owned-cache")
            os.environ["ZG_MONITOR_MODE"] = mode
            with Terminal(root, ["--ignored", "--exact", "pull::monitor::tests::monitor_child", "--nocapture"], width=150, height=40) as terminal:
                terminal.check_started()
                terminal.send(b"5", b"Paused", 15)
                terminal.send(b"s", b"Running", 15)
                terminal.send(b"p", b"Pause", 15)
                # Ratatui may emit only the final changed cell of "Paused".
                # Change views after the safe body boundary to read a newly
                # painted status, not the raw differential transition bytes.
                time.sleep(0.5)
                terminal.send(b"1", b"Quick", 15)
                terminal.send(b"5", b"Paused", 15)
                if mode == "tui":
                    terminal.send(b"s", b"Snapshot:", 20)
                else:
                    terminal.send(b"c", b"Cancel", 20)
                    time.sleep(0.5)
                    terminal.send(b"1", b"Quick", 15)
                    terminal.send(b"5", b"Cancelled", 15)
                os.write(terminal.master, b"q")
                terminal.finish()
                assert b"NOT_IN_OUTPUT" not in terminal.transcript
    print("Native download PTY: actual safe pause, resume/durable and cancel outcomes over one shared owner")
else:
    lifecycle(Path(sys.argv[2]), sys.argv[3])
