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
        terminal.send(b"\r", b"MODEL")
        terminal.send(b"served-fixture\r", b"Prompt")
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
        assert [model["id"] for model in models["data"]] == ["served-fixture"], models
        result = http("/completions", {"model": "served-fixture", "prompt": prompt,
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
    print("TUI cached lifecycle: quick run, repeated invoke, HTTP equality, navigation, decline quit, stop, unload/reload, owned listener/process retirement")


if len(sys.argv) == 2:
    with tempfile.TemporaryDirectory(prefix="zg-tui-") as tmp:
        shell(Path(tmp))
else:
    lifecycle(Path(sys.argv[2]), sys.argv[3])
