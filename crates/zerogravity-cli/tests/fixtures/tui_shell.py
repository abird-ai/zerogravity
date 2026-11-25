"""One real PTY lifecycle smoke test; no models, downloads or external network."""
import fcntl
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time

binary = str(Path(sys.argv[1]).resolve())

with tempfile.TemporaryDirectory(prefix="zg-tui-") as tmp:
    root = Path(tmp)
    env = dict(os.environ, HOME=tmp, XDG_CONFIG_HOME=str(root / "config"),
               XDG_CACHE_HOME=str(root / "cache"), TERM="xterm-256color", NO_COLOR="1")

    def resize(fd, width, height=28):
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))

    def exercise(args, actions, *, interactive=True, terminate=False):
        master, slave = pty.openpty()
        resize(slave, 100)
        initial = termios.tcgetattr(slave)
        proc = subprocess.Popen([binary, *args], cwd=root, env=env, stdin=slave,
                                stdout=slave, stderr=slave, start_new_session=True)
        pending = bytearray()
        transcript = bytearray()

        def until(needle):
            deadline = time.monotonic() + 10
            while needle not in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"missing {needle!r}: {transcript[-3000:]!r}")
                if select.select([master], [], [], remaining)[0]:
                    data = os.read(master, 65536)
                    pending.extend(data)
                    transcript.extend(data)
            del pending[:pending.index(needle) + len(needle)]

        try:
            if interactive:
                until(b"\x1b[?1049h")
                until(b"owners")
                assert termios.tcgetattr(slave) != initial
                for command, expected in actions:
                    if isinstance(command, int):
                        resize(slave, command)
                        os.kill(proc.pid, signal.SIGWINCH)
                    else:
                        os.write(master, command)
                    until(expected)
                if terminate:
                    proc.send_signal(signal.SIGTERM)
                else:
                    os.write(master, b"q")
                until(b"\x1b[?1049l")
            else:
                until(b"USAGE:")
                assert b"\x1b[?1049h" not in transcript
            assert proc.wait(timeout=10) == 0
            assert termios.tcgetattr(slave) == initial
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            os.close(slave)
            os.close(master)

    # Real navigation and modal typing, then narrow resize/nav and normal exit.
    exercise([], [(b"8", b"defaults"), (b":q", b"matching"),
                  (b"\x7flog\r", b"received."),
                  (60, b"ZeroGravity"), (b"\t", b"Navigation")])
    exercise([], [], terminate=True)
    exercise(["--help"], [], interactive=False)
    exercise(["--interactive", "never"], [], interactive=False)
    plain = subprocess.run([binary], cwd=root, env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, check=True)
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
