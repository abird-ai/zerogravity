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
import re

binary = str(Path(sys.argv[1]).resolve())


def environment(root):
    env = dict(os.environ, HOME=str(root), XDG_CONFIG_HOME=str(root / "config"),
               XDG_CACHE_HOME=str(root / "cache"), TERM="xterm-256color", NO_COLOR="1")
    env.pop("ZEROGRAVITY_API_KEY", None)
    return env


class Terminal:
    def __init__(self, root, args, width=100, height=28, hints=None):
        self.master, self.slave = pty.openpty()
        self.height = height
        self.repaint_width = width
        self.repaint_on_wait = False
        self.resize(width)
        self.initial = termios.tcgetattr(self.slave)
        env = environment(root)
        if hints:
            env.update(hints)
        self.proc = subprocess.Popen([binary, *args], cwd=root, env=env,
                                     stdin=self.slave, stdout=self.slave,
                                     stderr=self.slave, start_new_session=True)
        self.pending = bytearray()
        self.transcript = bytearray()

    def resize(self, width):
        self.width = width
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ,
                    struct.pack("HHHH", self.height, width, 0, 0))

    def until(self, needle, seconds=10, *, pattern=False):
        matcher = re.compile(needle if pattern else re.escape(needle))
        deadline = time.monotonic() + seconds
        next_repaint = time.monotonic() + 0.8
        while (match := matcher.search(self.pending)) is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"missing {needle!r}: {self.transcript[-4000:]!r}")
            # ANSI differential updates need not repeat the entire expected
            # word. A real resize forces a fresh cell paint without assuming
            # that an unchanged screen cell was emitted again.
            if self.repaint_on_wait and time.monotonic() >= next_repaint and self.proc.poll() is None and not needle.startswith(b"\x1b"):
                self.resize(self.repaint_width-1 if self.width == self.repaint_width else self.repaint_width)
                os.kill(self.proc.pid, signal.SIGWINCH)
                next_repaint = time.monotonic() + 0.8
            if select.select([self.master], [], [], min(remaining, 0.1))[0]:
                data = os.read(self.master, 65536)
                self.pending.extend(data)
                self.transcript.extend(data)
        del self.pending[:match.end()]

    def send(self, value, expected, seconds=10):
        self.pending.clear()
        while select.select([self.master], [], [], 0)[0]:
            self.transcript.extend(os.read(self.master, 65536))
        os.write(self.master, value)
        self.until(expected, seconds)

    def private_web_url(self):
        # The exact private capability is intentionally not retained in
        # ordinary PTY assertion transcripts.
        self.pending.clear()
        while select.select([self.master], [], [], 0)[0]:
            os.read(self.master, 65536)
        os.write(self.master, alt(b"b"))
        deadline = time.monotonic() + 10
        repaint = time.monotonic() + 0.8
        data = b""
        while time.monotonic() < deadline:
            if repaint is not None and time.monotonic() >= repaint:
                # A differential render may omit a character already
                # present under the URL. Force one full cell paint, without
                # copying the private capability into assertion transcripts.
                self.resize(149 if self.width == 150 else self.width + 1)
                os.kill(self.proc.pid, signal.SIGWINCH)
                repaint = None
            if select.select([self.master], [], [], 0.2)[0]:
                data = (data + os.read(self.master, 65536))[-65536:]
                match = re.search(rb"http://127\.0\.0\.1:\d+/ui/#control=[a-f0-9]{64}", data)
                if match:
                    return match.group().decode()
        raise AssertionError("TUI did not present a complete private Web URL")

    def check_started(self):
        self.until(b"\x1b[?1049h")
        self.until(b"Local runtime")
        assert termios.tcgetattr(self.slave) != self.initial

    def finish(self, *, interactive=True):
        if interactive:
            self.until(b"\x1b[?1049l")
            if b"\x1b[?1006h" in self.transcript:
                assert b"\x1b[?1006l" in self.transcript
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
                    os.write(terminal.master, b"\x11")
            else:
                terminal.until(b"USAGE:")
                assert b"\x1b[?1049h" not in terminal.transcript
            terminal.finish(interactive=interactive)

    exercise([], [(alt(b"8"), b"defaults"), (alt(b":") + b"q", b"matching"),
                  (b"\x7flog\r", b"received."),
                  (60, b"ZeroGravity"), (b"\t", b"Navigation")])
    exercise([], [], terminate=True)
    exercise(["--offline"], [(alt(b"4"), b"Cached"), (alt(b"m"), b"qwen3.5-plus"),
                              (alt(b"d"), b"prohibits"), (alt(b"5"), b"Download")])
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


# SGR mouse reports as emitted by a real terminal, using zero-based
# coordinates here and converting to the protocol's one-based cells.
def sgr(code, x, y, release=False):
    return f"\x1b[<{code};{x+1};{y+1}{'m' if release else 'M'}".encode()
def click(x,y):
    return sgr(0,x,y) + sgr(0,x,y,True)
def drag(x,y,end_x,end_y):
    return sgr(0,x,y) + sgr(32,end_x,end_y) + sgr(0,end_x,end_y,True)
def wheel(x,y,down=True):
    return sgr(65 if down else 64,x,y)


def alt(value):
    return b"\x1b" + value

def right_click(x,y):
    return sgr(2,x,y) + sgr(2,x,y,True)

def menu_cell(terminal,index,count):
    height = min(terminal.height-3,max(7,min(26,count+4)))
    return terminal.width//2, 2 + (terminal.height-3-height)//2 + index

def menu_action(terminal,index,count,expected,seconds=10,opener=b"\x1bOS"):
    terminal.send(opener,b"Actions")
    terminal.send(click(*menu_cell(terminal,index,count)),expected,seconds)

def filter_text(terminal,value,expected,seconds=10):
    terminal.send(b"\x06\x15" + value,expected,seconds)


def mouse_interaction(root):
    with Terminal(root, ["--offline"], width=150, height=24) as terminal:
        terminal.check_started()
        assert b"\x1b[?1006h" in terminal.transcript
        terminal.send(click(4,5), b"Cached")
        terminal.send(click(35,3), b"qwen3.5-plus")
        # The first visible row after a 3-line wheel movement is the fourth
        # catalog entry. Wheel doesn't alter selection; click must use the
        # actual scrolled row, not its index in the unscrolled catalog.
        terminal.send(wheel(30,10), b"gemma4")
        terminal.send(click(30,5), b"Qwen/Qwen3.8-27B")
        # Drag the rail from its default 22 cells to 29; clicking the newly
        # exposed rail area (previously workspace) must navigate.
        terminal.send(drag(22,18,29,18), b"Models")
        terminal.send(click(26,6), b"new acquisition disabled")
        terminal.send(click(26,5), b"Catalog")
        # Resize the actual inner divider, then select through the newly
        # enlarged list region (which was formerly the detail pane).
        terminal.send(drag(95,15,105,15), b"Selected")
        terminal.send(click(101,6), b"> qwen3.8-flash-next")
        menu_action(terminal,1,2,b"pinned-indexed-safetensors",opener=click(102,6))
        os.write(terminal.master,b"\x1b")
        time.sleep(0.2)
        # Maximize the list; select a row in the former details area, then
        # restore its old split and observe the selected exact source.
        terminal.send(click(102,4) + click(140,5), b"> qwen3.8-27b")
        terminal.send(click(147,4), b"Qwen/Qwen3.8-27B")
        # Focus details and return with keyboard equivalence. Opening a
        # modal masks every hit target behind it.
        terminal.send(b"\x1b[17~\x1bOP", b"Help")
        os.write(terminal.master, click(4,2))
        time.sleep(0.2)
        os.write(terminal.master,b"\x1b")
        time.sleep(0.2)
        terminal.send(alt(b"d"),b"prohibits")
        terminal.send(b"\x1b[19~", b"\x1b[?1006l")
        assert b"\x1b[?1006l" in terminal.transcript
        terminal.send(b"\x1b[19~", b"\x1b[?1006h")
        os.write(terminal.master,b"\x11")
        terminal.finish()
        assert b"\x1b[?1006l" in terminal.transcript

    # Compact vertical split: move the horizontal divider down to expose
    # an additional visible row and click the newly exposed row.
    with Terminal(root, ["--offline"], width=60, height=25) as terminal:
        terminal.check_started()
        terminal.send(click(2,0),b"Navigation")
        terminal.send(click(4,5),b"Cached")
        terminal.send(click(12,3),b"qwen3.5-plus")
        terminal.send(drag(25,13,25,14) + click(4,13),b"> gemma4-26b-a4b-it")
        # Narrow view collapses to one pane; Nav remains available.
        terminal.resize(30)
        os.kill(terminal.proc.pid,signal.SIGWINCH)
        terminal.until(b"ZeroGravity")
        terminal.send(click(2,0),b"Navigation")
        terminal.send(click(4,2),b"Local runtime")
        os.write(terminal.master,b"\x11")
        terminal.finish()
    assert not (root/"cache").exists()
    assert not (root/"config").exists()

    # Actual wrapped/raw detail viewport: wheel over detail and drag its
    # scrollbar without changing keyboard focus or the selected setting.
    document = root/"config/abird/zerogravity/config.yaml"
    document.parent.mkdir(parents=True)
    long_yaml = "\n".join(
        "# zzzzzzzzzzzzzzzzzzzz" if i == 15 else
        "# TAIL-TARGET-ONLY" if i == 39 else f"# filler-{i:02d}"
        for i in range(40)
    ) + "\nschema: zerogravity.pipeline.v2\n"
    document.write_text(long_yaml)
    with Terminal(root, [], width=100, height=24) as terminal:
        terminal.check_started()
        terminal.send(alt(b"8"), b"ui.color")
        terminal.send(alt(b"v"), b"filler-00")
        terminal.send(wheel(75,12), b"zzzzzzzzzzzzzzzzzzzz")
        terminal.send(drag(99,10,99,21), b"TAIL-TARGET-ONLY")
        terminal.send(wheel(30,10) + b"\r", b"Ctrl-U")
        os.write(terminal.master,b"\x1b")
        time.sleep(0.2)
        os.write(terminal.master,b"\x11")
        terminal.finish()
    assert document.read_text() == long_yaml

    # Explicit persistent opt-out is respected; temporary F8 override is
    # session-local and restoration still disables capture on exit.
    original = "schema: zerogravity.pipeline.v2\nui: {mouse: never}\n"
    document.write_text(original)
    with Terminal(root, []) as terminal:
        terminal.check_started()
        assert b"\x1b[?1006h" not in terminal.transcript
        terminal.send(b"\x1b[19~", b"\x1b[?1006h")
        os.write(terminal.master,b"\x11")
        terminal.finish()
        assert b"\x1b[?1006l" in terminal.transcript
    assert document.read_text() == original
    print("TUI mouse PTY: actual SGR click/wheel/drag, rail/inner split, maximize/restore, narrow nav, modal isolation, session toggle and persistent opt-out")


def mouse_controls(root):
    # A real foreground host, not simulated Action dispatch. The selected-row
    # menu, choice rows and modal buttons are all SGR clicks.
    profiles = root / "config/abird/zerogravity/profiles"
    desk = profiles / "desk.yaml"
    other = profiles / "other.yaml"
    with Terminal(root, [], width=150, height=40) as terminal:
        terminal.repaint_on_wait = True
        terminal.check_started()
        terminal.send(click(4,8), b"Profiles")
        terminal.until(b"> default")
        menu_action(terminal,1,3,b"lowercase",opener=right_click(30,5))
        terminal.send(b"desk" + click(105,24), b"Created", 10)
        assert desk.is_file()
        terminal.until(b"ui.theme", 10)
        terminal.send(click(33,1) + b"ui.theme", b"ui.theme = auto")
        menu_action(terminal,0,9,b"Enter apply",opener=right_click(35,6))
        original = desk.read_bytes()
        terminal.send(click(55,19), b"> Light")
        assert desk.read_bytes() == original
        os.write(terminal.master, click(81,23))
        time.sleep(0.2)
        menu_action(terminal,0,9,b"Enter apply",opener=right_click(35,6))
        terminal.send(click(55,18), b"> Dark")
        terminal.send(click(95,23), b"Saved", 10)
        assert "theme: dark" in desk.read_text()
        # The overflow menu includes the existing exact document action.
        terminal.until(b"ui.theme", 10)
        menu_action(terminal,4,9,b"Activated",10)
        assert desk.is_file()
        # Create a second profile, then exercise the explicit remove
        # confirmation (cancel first, confirm only on second invocation).
        terminal.send(click(4,8), b"desk")
        menu_action(terminal,1,3,b"lowercase",opener=right_click(30,5))
        terminal.send(b"other" + click(105,24), b"Created", 10)
        assert other.is_file()
        terminal.until(b"ui.theme",10)
        menu_action(terminal,7,9,b"y confirm")
        os.write(terminal.master, click(100,25))
        time.sleep(0.2)
        assert other.is_file()
        menu_action(terminal,7,9,b"y confirm")
        terminal.send(click(112,25), b"Removed",10)
        assert not other.exists() and desk.is_file()
        # A down on a valid button followed by release elsewhere must
        # not activate it. Existing keyboard navigation remains usable.
        terminal.send(b"\x1bOS",b"Actions")
        x,y=menu_cell(terminal,0,9)
        os.write(terminal.master,sgr(0,x,y)+sgr(0,5,4,True))
        os.write(terminal.master,b"\x1b")
        time.sleep(0.2)
        terminal.send(alt(b"9"), b"Operational logs")
        terminal.send(alt(b"1"), b"Quick")
        from web_shell import Client
        client = Client(terminal.private_web_url())
        os.write(terminal.master,b"\r")
        terminal.send(alt(b"8"),b"ui.theme")
        observation = client.config("@active")
        terminal.send(b"\x1bOS",b"Actions")
        x,y=menu_cell(terminal,0,9)
        os.write(terminal.master,sgr(0,x,y))
        time.sleep(0.25)
        client.action({"kind":"settings","action":{"command":"set",
            "snapshot":observation["snapshot"],"key":"ui.theme","value":"light"}})
        assert "theme: light" in desk.read_text()
        time.sleep(0.25)
        os.write(terminal.master,sgr(0,x,y,True))
        time.sleep(0.2)
        os.write(terminal.master,b"\x1b")
        time.sleep(0.2)
        terminal.send(alt(b"9"),b"Operational logs")
        terminal.send(b"\x11",b"y confirm")
        os.write(terminal.master,click(112,25))
        terminal.finish()
        client.closed()
    with Terminal(root,[],width=50,height=18) as terminal:
        terminal.check_started()
        terminal.send(b"\x1bOS",b"> Run  (Enter)")
        terminal.send(b"\r",b"MODEL or ID=MODEL")
        os.write(terminal.master,b"\x1b")
        time.sleep(0.2)
        terminal.send(click(terminal.width-4,1),b"> Run  (Enter)")
        os.write(terminal.master,b"\x1b")
        time.sleep(0.2)
        menu_action(terminal,1,4,b"ui.theme",opener=click(terminal.width-12,terminal.height-1))
        os.write(terminal.master,b"\x11")
        terminal.finish()
    assert not (root/"cache").exists()
    assert not (profiles.parent/"config.yaml").exists()
    print("TUI action PTY: mouse create/edit/theme preview/cancel/save, overflow activate, remove cancel/confirm, release mismatch, stale Web edit, narrow overflow; no cache effects")


def editor_and_choices(root):
    # Real mouse input into the same native configuration draft and the
    # same local model/log observations. No cache/model acquisition.
    document = root / "config/abird/zerogravity/config.yaml"
    document.parent.mkdir(parents=True)
    original = "schema: zerogravity.pipeline.v2\nrun:\n  backend: cpu\n  task: {raw: false}\n"
    document.write_text(original)
    with Terminal(root, [], width=150, height=40) as terminal:
        terminal.repaint_on_wait = True
        terminal.check_started()
        terminal.send(alt(b"8"), b"ui.theme")
        terminal.send(click(33,1)+b"run.backend",b"run.backend = cpu")
        menu_action(terminal,0,9,b"> cpu",opener=right_click(35,6))
        terminal.send(click(51,19),b"> rocm")
        assert document.read_text()==original
        os.write(terminal.master,click(86,22))
        time.sleep(0.2)
        menu_action(terminal,0,9,b"> cpu",opener=right_click(35,6))
        terminal.send(click(51,19),b"> rocm")
        terminal.send(click(99,22),b"Saved",10)
        assert "backend: rocm" in document.read_text()
        terminal.until(b"run.backend = rocm")
        filter_text(terminal,b"run.task.raw",b"run.task.raw = false")
        menu_action(terminal,0,9,b"> false",opener=right_click(35,6))
        terminal.send(click(51,17),b"> true")
        terminal.send(click(99,21),b"Saved",10)
        assert "raw: true" in document.read_text()
        terminal.until(b"run.task.raw = true")
        # A finite choice uses the same exact document bytes as a YAML
        # value. An external update cannot be overwritten by its Apply.
        menu_action(terminal,0,9,b"> true",opener=right_click(35,6))
        terminal.send(click(51,16),b"> false")
        concurrent = "# concurrent edit\n" + document.read_text()
        document.write_text(concurrent)
        terminal.send(click(99,21),b"changed",10)
        assert document.read_text()==concurrent
        os.write(terminal.master,b"\x1b")
        time.sleep(0.2)
        # Log filter choices are local, not configuration writes.
        before = document.read_bytes()
        terminal.send(alt(b"9"),b"Operational logs")
        menu_action(terminal,1,3,b"All levels")
        terminal.send(click(51,19),b"> ERROR")
        terminal.send(click(99,21),b"returned")
        menu_action(terminal,2,3,b"All components")
        terminal.send(click(51,15),b"> action")
        terminal.send(click(99,23),b"action")
        assert document.read_bytes()==before

        # Search the existing catalog locally and replace the visible
        # query via a drag selection, not a remote provider query.
        terminal.send(alt(b"4"),b"Cached")
        terminal.send(click(35,3),b"qwen3.5-plus")
        # Ordinary input immediately filters, including former shortcuts.
        terminal.send(b"gemma4",b"> gemma4-e2b-it")
        terminal.send(b"q8:/?",b"No matches")
        # Escape clears the active filter. No operation was submitted.
        terminal.send(b"\x1b",b"qwen3.5-plus")
        terminal.send(b"gemma4",b"> gemma4-e2b-it")
        x=31
        terminal.send(drag(x,1,x+6,1)+b"qwen3.5",b"> qwen3.5-plus")
        terminal.send(alt(b"m"),b"Partial")
        terminal.send(alt(b"m"),b"Cached")
        terminal.send(alt(b"m"),b"Catalog")
        terminal.until(b"qwen3.5-plus")
        os.write(terminal.master,b"\x11")
        terminal.finish()

    # A long raw editor is independently scrollable. Select wide and
    # combining graphemes by their painted cells; original YAML bytes
    # outside the selected text must be unchanged.
    long_yaml = "# a界e\u0301z\n" + "".join(f"# filler-{i:02d}\n" for i in range(65)) + original
    document.write_text(long_yaml)
    with Terminal(root, [], width=150, height=40) as terminal:
        terminal.repaint_on_wait = True
        terminal.check_started()
        terminal.send(alt(b"8"),b"ui.theme")
        terminal.send(alt(b"w"),b"Ctrl-S")
        terminal.send(click(terminal.width-2,4), b"filler-00")
        terminal.send(drag(4,4,7,4) + b"X",b"Xz")
        terminal.send(click(5,4) + b"_",b"_z")
        terminal.send(click(terminal.width-8,36),b"Saved",10)
        assert document.read_text() == long_yaml.replace("a界e\u0301z","aX_z",1)
        os.write(terminal.master,b"\x11")
        terminal.finish()
    assert not (root/"cache").exists()
    print("TUI editor/choices PTY: typed effective enum and boolean, cancel/apply, log pickers, local shelf search, grapheme drag/caret replacement and independent raw scroll; exact saved bytes")


def compact_ergonomics(root):
    with Terminal(root,[],width=20,height=7) as terminal:
        terminal.repaint_on_wait=True
        terminal.repaint_width=21
        # The body is too short for the normal modal; the full-screen
        # compact menu must still expose a visible action and close.
        terminal.until(b"\x1b[?1049h")
        terminal.send(b"\x1bOS",b"> Run")
        terminal.send(click(4,1),b"Apply")
        os.write(terminal.master,b"\x1b")
        time.sleep(0.2)
        terminal.send(alt(b":"),b"Go to")
        terminal.send(b"\x1b[B"*8,b"> Logs")
        terminal.send(b"\r",b"Search")
        terminal.send(b"\x1bOP",b"Help")
        terminal.send(b"\x1b[4~",b"quit")
        os.write(terminal.master,b"\x1b")
        time.sleep(0.2)
        os.write(terminal.master,b"\x11")
        terminal.finish()
    assert not (root/"cache").exists() and not (root/"config").exists()
    print("TUI compact PTY: 20x7 context action, scrolled destination palette and help, no hidden effects")


def appearance(root):
    default = root / "config/abird/zerogravity/config.yaml"
    hints = {"COLORTERM": "truecolor", "COLORFGBG": "0;15", "NO_COLOR": ""}
    with Terminal(root, [], width=100, height=28, hints=hints) as terminal:
        terminal.repaint_on_wait = True
        terminal.check_started()
        terminal.until(b"Quick")
        assert b"48;2;255;255;255" in terminal.transcript
        terminal.send(alt(b"T"), b"ui.theme = auto")
        terminal.send(b"\r", b"Enter apply")
        terminal.send(b"j", b"> Dark")
        assert b"48;2;23;26;32" in terminal.transcript
        assert not default.exists()
        terminal.send(b"\x1b", b"ui.theme = auto")
        assert not default.exists()
        terminal.send(b"\r", b"Enter apply")
        terminal.send(b"jj", b"> Light")
        terminal.send(b"\r", b"Saved", 10)
        assert "theme: light" in default.read_text()
        terminal.send(alt(b"1"), b"Quick")
        from web_shell import Client
        client = Client(terminal.private_web_url())
        os.write(terminal.master, b"\r")
        assert client.snapshot()["configuration"]["theme"] == "light"
        # Preview of an exact observed document cannot overwrite a newer
        # setting applied by the attached Web client.
        terminal.send(alt(b"T"), b"ui.theme = light")
        terminal.send(b"\r", b"Enter apply")
        document = client.config("@active")
        client.action({"kind":"settings", "action":{"command":"set",
            "snapshot":document["snapshot"], "key":"ui.theme", "value":"dark"}})
        assert client.snapshot()["configuration"]["theme"] == "dark"
        assert "theme: dark" in default.read_text()
        time.sleep(0.4)  # allow the foreground observation of the completed action
        terminal.send(b"\r", b"changed", 10)
        assert "theme: dark" in default.read_text()
        terminal.send(b"\x1b", b"48;2;23;26;32", 10)
        # Inactive profile theme edits must not reconfigure active appearance.
        client.action({"kind":"settings","action":{"command":"create","name":"theme-alt"}})
        alternate = client.config("theme-alt")
        client.action({"kind":"settings","action":{"command":"set",
            "snapshot":alternate["snapshot"],"key":"ui.theme","value":"light"}})
        assert client.snapshot()["configuration"]["theme"] == "dark"
        assert "theme: light" in (default.parent / "profiles/theme-alt.yaml").read_text()
        terminal.send(b"\x11", b"confirm")
        os.write(terminal.master, b"y")
        terminal.finish()
        client.closed()
    # Reopen, verify explicit mode wins over the Auto light hint.
    with Terminal(root, [], hints=hints) as terminal:
        terminal.check_started()
        assert b"48;2;23;26;32" in terminal.transcript
        os.write(terminal.master, b"\x11")
        terminal.finish()
    # Restricted terminals preserve usable text, and color=never wins.
    for args, term, color, expected in [
        ([], "xterm-256color", "", b"48;5;"),
        ([], "xterm", "", b"48;5;0m"),
        (["--color", "never"], "xterm-256color", "truecolor", None),
    ]:
        with Terminal(root, args, hints={"TERM":term, "COLORTERM":color, "NO_COLOR":""}) as terminal:
            terminal.check_started()
            if expected:
                assert expected in terminal.transcript, terminal.transcript[-2000:]
            else:
                assert not re.search(rb"\x1b\[(?:38|48);", terminal.transcript)
            os.write(terminal.master, b"\x11")
            terminal.finish()
    assert not (root / "cache").exists()
    print("TUI appearance PTY: Auto light, Dark/Light preview, cancel/no write, explicit save, stale Web edit refusal, inactive profile isolation, persisted reopen, 256/16/no-color")


def settings_lifecycle(root):
    root.mkdir()
    profiles = root / "config/abird/zerogravity/profiles"
    profiles.mkdir(parents=True)
    desk = profiles / "desk.yaml"
    copied = profiles / "copy.yaml"
    broken = profiles / "broken.yaml"
    broken.write_text("schema: zerogravity.pipeline.v2\nunknown_field: true\n")
    valid = "schema: zerogravity.pipeline.v2\nrun:\n  backend: cpu\n  task: {context: 128, max_tokens: 8}\nui: {color: never}\n"
    paste = lambda text: b"\x1b[200~" + text.encode() + b"\x1b[201~"

    with Terminal(root, [], width=150, height=40) as terminal:
        terminal.repaint_on_wait = True
        terminal.check_started()
        terminal.send(alt(b"7"), b"broken")
        terminal.send(alt(b"c"), b"lowercase")
        terminal.send(b"desk\r", b"Created")
        assert desk.is_file()
        terminal.until(b"run.backend", 10)
        filter_text(terminal,b"run.backend",b"> run.backend")
        terminal.send(b"\r", b"Ctrl-U")
        terminal.send(b"cpu\r", b"Saved", 10)
        assert "backend: cpu" in desk.read_text()
        # Validation refusal retains the old document and reopens the draft.
        original = desk.read_bytes()
        terminal.until(b"run.backend", 10)
        terminal.send(b"\r", b"Ctrl-U")
        terminal.send(b"\x15invalid-backend\r", b"failed", 10)
        assert desk.read_bytes() == original
        os.write(terminal.master, b"\x1b")
        time.sleep(0.2)  # allow standalone Escape to be decoded before the next key
        # The exact observed file is captured before editing, not reopened at
        # submit time; a cooperating external change must not be overwritten.
        terminal.send(b"\r", b"Ctrl-U")
        desk.write_text(valid.replace("backend: cpu", "backend: rocm"))
        terminal.send(b"\x15cpu\r", b"changed", 10)
        assert desk.read_text() == valid.replace("backend: cpu", "backend: rocm")
        os.write(terminal.master, b"\x1b")
        time.sleep(0.2)  # allow standalone Escape to be decoded before the next key
        terminal.send(alt(b"r"), b"rocm", 10)
        terminal.send(alt(b"w"), b"Ctrl-S")
        terminal.send(b"\x15" + paste(valid) + b"\x13", b"Saved", 10)
        assert desk.read_text() == valid
        terminal.until(b"run.backend", 10)
        terminal.send(alt(b"a"), b"Activated", 10)
        terminal.send(alt(b"x"), b"another", 10)
        assert desk.is_file()
        terminal.send(alt(b"y"), b"preserved")
        terminal.send(b"copy\r", b"Created", 10)
        assert copied.read_text() == valid
        terminal.until(b"run.backend", 10)
        terminal.send(alt(b"x"), b"confirm")
        os.write(terminal.master, b"n")
        assert copied.exists()
        terminal.send(alt(b"x"), b"confirm")
        terminal.send(b"y", b"Removed", 10)
        assert not copied.exists() and desk.exists()
        # Repair an invalid named document using the same full raw YAML owner.
        terminal.send(alt(b"7"), b"broken")
        terminal.send(b"\x1b[B\r", b"Invalid", 10)
        terminal.send(alt(b"w"), b"Ctrl-S")
        terminal.send(b"\x15" + paste(valid) + b"\x13", b"Saved", 10)
        assert broken.read_text() == valid
        # The Logs surface records actions/owner outcomes, not draft contents.
        terminal.send(alt(b"9"), b"Operational")
        terminal.send(b"\x1b[A",b"Paused")
        terminal.send(b"\r",b"Correlation:")
        terminal.until(b"Sequence:")
        os.write(terminal.master,b"\x1b")
        time.sleep(0.2)
        terminal.send(b"\x1b[4~",b"Following")
        terminal.send(alt(b"p"), b"Paused")
        filter_text(terminal,"Save config".encode(),b"Save")
        terminal.send(alt(b"p"), b"Following")
        terminal.send(alt(b"1"), b"Quick")
        from web_shell import Client
        client = Client(terminal.private_web_url())
        os.write(terminal.master, b"\r")
        native_logs = client.inventory("logs")
        assert any(row["message"] == "Save config completed" for row in native_logs["entries"])
        assert "invalid-backend" not in json.dumps(native_logs)
        assert "broken" in client.inventory("profiles")["profiles"]
        terminal.send(alt(b"8"), b"run.backend")
        filter_text(terminal,b"run.backend",b"> run.backend")
        document = client.config("broken")
        client.action({"kind": "settings", "action": {"command": "set", "snapshot": document["snapshot"],
                      "key": "run.backend", "value": "rocm"}})
        terminal.until(b"rocm", 10)
        assert "backend: rocm" in broken.read_text()
        document = client.config("broken")
        client.action({"kind": "settings", "action": {"command": "activate", "snapshot": document["snapshot"]}})
        assert client.snapshot()["configuration"]["profile"] == "broken"
        terminal.send(alt(b"9"), b"Operational")
        filter_text(terminal,"Activate profile".encode(),b"Activate profile")
        filter_text(terminal,"Validate config".encode(),b"notifications")
        # No navigation, resize, user input or runtime event after the Web
        # validation. The shared log revision alone must repaint the TUI.
        terminal.repaint_on_wait = False
        client.action({"kind": "settings", "action": {"command": "validate", "snapshot": document["snapshot"]}})
        terminal.until(b"completed", 10)
        assert any(row["message"] == "Validate config completed" for row in client.inventory("logs")["entries"])
        terminal.repaint_on_wait = True
        terminal.send(b"\x11", b"confirm")
        os.write(terminal.master, b"y")
        terminal.finish()
        client.closed()
    assert not (root / "cache").exists()
    assert not (profiles.parent / "config.yaml").exists()
    print("TUI settings PTY: create, search/edit, invalid/stale refusal, raw repair, activate/protect, copy, declined/confirmed remove, operational log filters")


def lifecycle(root, expected):
    # Independently exercise the actual browser before the cross-client
    # PTY flow, so a terminal repaint issue does not mask that evidence.
    if os.environ.get("ZG_TEST_WEB") == "1" and os.environ.get("ZG_TEST_BROWSER") == "1":
        smoke = Path(__file__).resolve().parents[2] / "web/src/browser-smoke.mjs"
        subprocess.run(["node", str(smoke), binary, "--resident", str(root), expected],
                       env=environment(root), cwd=root, check=True)
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
        terminal.repaint_on_wait = os.environ.get("ZG_TEST_WEB") == "1"
        terminal.send(alt(b"6"), b"cpu:0", 30)
        # Original inventory ordering ends in host memory. Use the original
        # read-only CLI observation to select that exact physical row.
        observation = subprocess.run([binary, "device", "list", "--format=json"], cwd=root,
                                     env=environment(root), capture_output=True, check=True)
        devices = json.loads(observation.stdout)["devices"]
        cpu_index = next(i for i, device in enumerate(devices) if device["id"] == "cpu:0")
        if cpu_index:
            os.write(terminal.master, b"\x1b[B" * cpu_index)
        menu_action(terminal,0,3,b"MODEL")
        terminal.send(click(105,24), b"accepted", 60)
        terminal.send(alt(b"4"), b"fixture/serve")
        # Exact snapshot identity is disclosed only on request.
        menu_action(terminal,2,9,b"sha256:",opener=right_click(30,5))
        os.write(terminal.master,b"\x1b")
        time.sleep(0.2)
        def model_action(index,expected,seconds=10):
            menu_action(terminal,index,9,expected,seconds,opener=right_click(30,5))
        model_action(3,b"checked;")
        model_action(4,b"Payloads",60)
        model_action(5,b"replacement.")
        terminal.send(b"tui-fixture" + click(105,24), b"created", 30)
        # Alias publication triggers a cold inventory refresh; let the
        # new observation settle before pressing another exact-row action.
        time.sleep(0.5)
        model_action(8,b"confirm")
        os.write(terminal.master, b"n")
        model_action(1,b"Prompt")
        # Original profile raw=true. TUI completion must be actual output.
        terminal.send(b"\x1b[200~" + prompt.encode() + b"\x1b[201~" + click(105,24), b"length", 90)
        assert b"Response" in terminal.transcript
        terminal.send(alt(b"2"), b"registration")
        terminal.send(b"\x1bOR",b"capacity:")
        terminal.send(b"\x1bOR",b"Response")
        terminal.send(alt(b"p"), b"Prompt")
        terminal.send(b"\x1b[200~" + prompt.encode() + b"\x1b[201~\r", b"length", 30)
        terminal.send(alt(b"3"), b"OpenAI-compatible")
        menu_action(terminal,0,1,b"IP:PORT",opener=click(terminal.width-4,1))
        terminal.send(address.encode() + click(105,24), b"Listening:", 30)
        models = http("/models")
        assert len(models["data"]) == 1, models
        loaded_alias = models["data"][0]["id"]
        assert len(loaded_alias) == 64 and all(c in "0123456789abcdef" for c in loaded_alias), models
        result = http("/completions", {"model": loaded_alias, "prompt": prompt,
                                       "max_tokens": 2, "stream": False})
        assert result["choices"][0]["text"] == expected, result
        assert result["usage"]["completion_tokens"] > 0
        client = None
        if os.environ.get("ZG_TEST_WEB") == "1":
            from web_shell import Client
            client = Client(terminal.private_web_url())
            observed = client.snapshot()
            registered = observed["runtime"]["models"]
            assert len(registered) == 1
            assert registered[0]["aliases"] == [loaded_alias]
            assert isinstance(registered[0]["registration_id"], str)
            registration = registered[0]["registration_id"]
            generation = observed["serving"]["generation"]
            assert observed["serving"]["address"] == address
            assert observed["configuration"]["profile"] == "local"
            # Control capability does not authorize inference or start a new
            # runtime. Closing the URL modal is only presentation.
            assert client.request("/v1/models")[0] == 404
            assert observed["web_action"] is None  # native inference is private
            os.write(terminal.master, b"\r")
            cached = client.inventory("models")["cached"]
            assert len(cached) == 1 and cached[0]["digest"] == loaded_alias
            # A loaded resident may release its cache readers after materialization.
            # Do not assume resident ownership is a cache-retirement lock.
            checked = client.action({"kind": "inspect", "digest": loaded_alias})
            assert "checked" in checked["message"].lower()
            assert client.snapshot()["runtime"]["models"][0]["registration_id"] == registration
            generated = client.action({"kind": "invoke", "alias": loaded_alias,
                                       "registration_id": registration, "prompt": prompt, "chat": False})
            output = generated["response"]
            assert output["text"] == expected
            assert output["owner"] == {"alias": loaded_alias, "registration_id": registration}
            assert output["usage"]["completion_tokens"] == result["usage"]["completion_tokens"]
            assert output["usage"]["prompt_tokens"] == result["usage"]["prompt_tokens"]
            assert client.snapshot()["runtime"]["models"][0]["registration_id"] == registration
            client.post({"kind": "cancel", "id": generated["id"]}, status=409)
            # Both editors retain exact observed bytes, not a writable alias
            # to the active runtime. Change future policy while an admitted
            # registration and the original listener remain live.
            stale = client.config("local")
            assert stale["active"] and stale["raw"] is None
            admitted_context = observed["runtime"]["scheduler"]["context"]
            terminal.send(alt(b"8"), b"run.backend")
            filter_text(terminal,b"run.task.context",b"> run.task.context")
            terminal.send(b"\r", b"Ctrl-U")
            with client.events() as events:
                terminal.send(b"\x15128\r", b"Saved", 15)
                for _ in range(24):
                    if events.readline() == b"event: configuration\n":
                        break
                else:
                    raise AssertionError("native edit did not invalidate browser configuration")
            def set_context(snapshot, value):
                return {"kind": "settings", "action": {"command": "set", "snapshot": snapshot["snapshot"],
                        "key": "run.task.context", "value": value}}
            client.action(set_context(stale, "96"), state="failed")
            fresh = client.config("local")
            assert next(row for row in fresh["rows"] if row["key"] == "run.task.context")["value"].strip() == "128"
            assert client.snapshot()["runtime"]["scheduler"]["context"] == admitted_context
            client.action(set_context(fresh, "64"))
            terminal.until(b"run.task.context = 64", 15)
            observed = client.snapshot()
            assert observed["runtime"]["models"][0]["registration_id"] == registration
            assert observed["serving"]["generation"] == generation
            assert observed["runtime"]["scheduler"]["context"] == admitted_context
            assert http("/models") == models
            replay = http("/completions", {"model": loaded_alias, "prompt": prompt,
                                           "max_tokens": 2, "stream": False})
            assert replay["choices"][0]["text"] == expected
            # Both origins' outcomes share one bounded ring, never raw drafts
            # or native response text. A rejected stale mutation is a failure.
            log = client.inventory("logs")["entries"]
            assert any(row["message"] == "Save config returned error" for row in log)
            assert any(row["message"] == "Save config completed" for row in log)
            print("Shared admitted settings: native edit, stale Web refusal, Web restore observed by native, unchanged scheduler/registration/listener and exact HTTP output")
        # Activate a different startup profile without reconfiguring the
        # existing listener/scheduler. Restore the original for later load.
        terminal.repaint_on_wait = True
        terminal.send(alt(b"9"), b"accepted")
        terminal.send(alt(b"7"), b"default")
        terminal.send(b"\r", b"defaults")
        if client:
            with client.events() as events:
                terminal.send(alt(b"a"), b"Activated")
                assert events.readline() == b"event: refresh\n"
                assert client.snapshot()["configuration"]["profile"] == "default"
        else:
            terminal.send(alt(b"a"), b"Activated")
        assert http("/models") == models
        terminal.send(alt(b"7"), b"local")
        terminal.send(b"\x1b[B\r", b"run.backend")
        terminal.send(alt(b"a"), b"Activated")
        assert http("/models") == models
        if client:
            observed = client.snapshot()
            assert observed["configuration"]["profile"] == "local"
            assert observed["runtime"]["models"][0]["registration_id"] == registration
            assert observed["serving"]["generation"] == generation
        terminal.repaint_on_wait = client is not None
        # Navigation and a declined quit leave the same listener alive.
        terminal.send(alt(b"1"), b"Quick")
        terminal.send(b"\x11", b"confirm")
        os.write(terminal.master, b"n")
        assert http("/models") == models
        terminal.send(alt(b"3"), b"OpenAI-compatible")
        menu_action(terminal,0,1,b"confirm",opener=click(terminal.width-4,1))
        if client:
            with client.events() as events:
                terminal.send(click(112,25), b"Listener stopped", 30)
                assert events.readline() == b"event: refresh\n"
                observed = client.snapshot()
                assert observed["serving"]["state"] == "idle"
                assert observed["runtime"]["models"][0]["registration_id"] == registration
        else:
            terminal.send(click(112,25), b"Listener stopped", 30)
        closed()
        terminal.send(alt(b"2"), b"registered")
        if client:
            terminal.repaint_on_wait = True
            retired = client.action({"kind": "unload", "alias": loaded_alias,
                                     "registration_id": registration, "confirm": True})
            assert retired["state"] == "completed"
            client.observed(lambda row: row["runtime"]["status"]["loaded_owners"] == 0)
            terminal.send(alt(b"1"), b"Quick")
            # The Web load-and-run uses the configured alias and original
            # resident plan, not a second service. No browser connection owns
            # the worker once the POST has returned its exact id.
            accepted = client.post({"kind": "run_source", "source": "", "prompt": prompt, "chat": False})
            generated = client.result(accepted["id"])
            assert generated["response"]["text"] == expected
            second = client.snapshot()["runtime"]["models"][0]
            assert second["aliases"] == ["configured"] and second["registration_id"] != registration
            assert generated["response"]["owner"]["registration_id"] == second["registration_id"]
            terminal.send(alt(b"2"), b"configured", 30)
            # Stale selected owner cannot retire a replacement even when the
            # alias is supplied from a fresh observation.
            client.action({"kind": "unload", "alias": "configured", "registration_id": registration,
                           "confirm": True}, state="failed")
            assert client.snapshot()["runtime"]["models"][0]["registration_id"] == second["registration_id"]
            client.action({"kind": "start", "address": address})
            observed = client.observed(lambda row: row["serving"]["state"] == "listening")
            assert observed["serving"]["generation"] != generation
            terminal.send(alt(b"3"), address.encode(), 30)
            assert [model["id"] for model in http("/models")["data"]] == ["configured"]
            client.action({"kind": "stop", "generation": generation, "confirm": True}, state="failed")
            assert client.snapshot()["serving"]["generation"] == observed["serving"]["generation"]
            # Native invocation and Web observation still use the same
            # registered owner. Browser receipts are not overwritten by it.
            terminal.send(alt(b"2"), b"configured")
            terminal.send(alt(b"p"), b"Prompt")
            terminal.send(b"\x1b[200~" + prompt.encode() + b"\x1b[201~\r", b"length", 30)
            terminal.send(alt(b"3"), address.encode(), 30)
        else:
            terminal.send(alt(b"u"), b"confirm")
            terminal.send(b"y", b"retired", 60)
            # The last retirement makes the same runtime available for a new load.
            terminal.send(b"l", b"MODEL")
            terminal.send(b"\r", b"loaded", 90)
            terminal.send(alt(b"3"), b"OpenAI-compatible")
            terminal.send(b"s", b"IP:PORT")
            terminal.send(address.encode() + b"\r", b"Listening:", 30)
            assert [model["id"] for model in http("/models")["data"]] == ["configured"]
        terminal.send(b"\x11", b"confirm")
        os.write(terminal.master, b"y")
        terminal.finish()
        closed()
        if client:
            client.closed()
            print("TUI-attached Web: real invoke/unload/load-and-run/start, exact stale registration/generation refusal, profile/owner invalidations and joined retirement")
    alias = subprocess.run([binary, "model", "inspect", "by-name:tui-fixture", "--format=json"],
                           cwd=root, env=environment(root), capture_output=True, check=True)
    assert json.loads(alias.stdout)["snapshot"] == loaded_alias
    with Terminal(root, ["--profile", "local"], width=150, height=40) as terminal:
        terminal.check_started()
        if os.environ.get("ZG_TEST_WEB") == "1":
            from web_shell import Client
            client = Client(terminal.private_web_url())
            os.write(terminal.master, b"\r")
            terminal.send(alt(b"4"), b"fixture/serve")
            preview = client.action({"kind": "removal_preview", "digest": loaded_alias})
            assert "no mutation" in preview["message"]
            assert len(client.inventory("models")["cached"]) == 1
            with client.events() as events:
                removed = client.action({"kind": "remove", "digest": loaded_alias, "confirm": True})
                assert "durable" in removed["message"]
                for _ in range(20):
                    if events.readline() == b"event: models\n":
                        break
                else:
                    raise AssertionError("missing cache invalidation")
            assert client.inventory("models")["cached"] == []
            terminal.repaint_on_wait = True
            # A full paint may move the cursor across the blank between
            # the count and label, rather than emitting a literal space.
            terminal.until(rb"0(?: |\x1b\[\d+;\d+H)models", 15, pattern=True)
        else:
            terminal.send(alt(b"4"), b"fixture/serve")
            terminal.send(alt(b"X"), b"preview;", 30)
            terminal.send(alt(b"x"), b"confirm")
            terminal.send(b"y", b"durable", 60)
        if os.environ.get("ZG_TEST_WEB") == "1":
            terminal.send(b"\x11", b"confirm")
            os.write(terminal.master, b"y")
        else:
            os.write(terminal.master, b"\x11")
        terminal.finish()
        if os.environ.get("ZG_TEST_WEB") == "1":
            client.closed()
    listing = subprocess.run([binary, "model", "list", "--format=json"], cwd=root,
                             env=environment(root), capture_output=True, check=True)
    assert json.loads(listing.stdout) == []
    print("TUI cached lifecycle: inspect/verify, alias, declined removal, exact-digest run, repeated invoke, HTTP equality, active profile switching retains listener, stop/unload/reload/quit, preview and durable removal")


def shared_download():
    from web_shell import Client
    smoke = Path(__file__).resolve().parents[2] / "web/src/browser-smoke.mjs"

    def browser_line(browser, expected):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            assert select.select([browser.stdout], [], [], max(0, deadline - time.monotonic()))[0], expected
            line = browser.stdout.readline()
            assert line, "attached browser exited before " + expected
            if line.decode().strip() == expected:
                return
        raise AssertionError("attached browser did not report " + expected)

    for mode, outcome in (("tui", "durable"), ("tui-cancel", "cancelled")):
        with tempfile.TemporaryDirectory(prefix="zg-shared-download-") as tmp:
            root = Path(tmp)
            os.environ["ZG_MONITOR_FIXTURE"] = str(root / "cache/abird/zerogravity/models")
            os.environ["ZG_MONITOR_MODE"] = mode
            with Terminal(root, ["--ignored", "--exact", "pull::monitor::tests::monitor_child", "--nocapture"],
                          width=150, height=40) as terminal:
                terminal.check_started()
                terminal.repaint_on_wait = True
                private_url = terminal.private_web_url()
                client = Client(private_url)
                os.write(terminal.master, b"\r")
                identity = client.observed(lambda row: row["download"]["status"] == "paused")["download"]["id"]
                # Feed the private capability over the owned pipe, not argv
                # or diagnostics. The actual Solid UI uses SSE and typed POST.
                with subprocess.Popen(["node", str(smoke), binary, "--attached-download", outcome],
                                      cwd=root, env=environment(root), stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, bufsize=0) as browser:
                    try:
                        browser.stdin.write((private_url + "\n").encode())
                        browser.stdin.flush()
                        browser_line(browser, "Attached browser ready")
                        terminal.send(alt(b"5"), b"Paused")
                        terminal.send(b"s", b"Running")
                        client.observed(lambda row: row["download"]["total_bytes"] > 0)
                        terminal.send(b"p", b"Pause")
                        parked = client.observed(lambda row: row["download"]["status"] == "paused")["download"]
                        assert parked["id"] == identity and parked["checkpoint_bytes"] > 0
                        browser.stdin.write(b"native-paused\n")
                        browser.stdin.flush()
                        browser_line(browser, "Attached browser terminal")
                        terminal.until(b"Snapshot:" if outcome == "durable" else b"Cancelled", 15)
                        assert browser.wait(timeout=15) == 0
                        ended = client.snapshot()["download"]
                        assert ended["id"] == identity and ended["status"] == outcome and ended["terminal"]
                        assert client.snapshot()["action"] is None
                        inventory = client.inventory("models")
                        if outcome == "durable":
                            assert inventory["cached"][0]["digest"] == ended["snapshot"]
                        else:
                            assert inventory["cached"] == [] and inventory["partial"][0]["bytes"] > 0
                        client.post({"kind": "download", "operation_id": identity, "command": "resume"}, status=409)
                        # The browser closed independently; native quit joins
                        # the attached host and the original transfer worker.
                        terminal.send(b"\x11", b"confirm")
                        os.write(terminal.master, b"y")
                        terminal.finish()
                        client.closed()
                    finally:
                        if browser.poll() is None:
                            browser.terminate()
                            try:
                                browser.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                browser.kill()
                                browser.wait()
                assert b"NOT_IN_OUTPUT" not in terminal.transcript
                print("Shared transfer:", outcome, "native safe pause, attached Firefox command, same operation and joined retirement")


if len(sys.argv) == 2:
    with tempfile.TemporaryDirectory(prefix="zg-tui-") as tmp:
        shell(Path(tmp))
        settings_lifecycle(Path(tmp) / "settings")
elif sys.argv[2] == "--settings-fixture":
    with tempfile.TemporaryDirectory(prefix="zg-tui-settings-") as tmp:
        settings_lifecycle(Path(tmp) / "settings")
elif sys.argv[2] == "--mouse-fixture":
    with tempfile.TemporaryDirectory(prefix="zg-tui-mouse-") as tmp:
        mouse_interaction(Path(tmp))
elif sys.argv[2] == "--controls-fixture":
    with tempfile.TemporaryDirectory(prefix="zg-tui-controls-") as tmp:
        mouse_controls(Path(tmp))
elif sys.argv[2] == "--editor-fixture":
    with tempfile.TemporaryDirectory(prefix="zg-tui-editor-") as tmp:
        editor_and_choices(Path(tmp))
elif sys.argv[2] == "--ergonomics-fixture":
    with tempfile.TemporaryDirectory(prefix="zg-tui-ergonomics-") as tmp:
        compact_ergonomics(Path(tmp))
elif sys.argv[2] == "--appearance-fixture":
    with tempfile.TemporaryDirectory(prefix="zg-tui-appearance-") as tmp:
        appearance(Path(tmp))
elif sys.argv[2] == "--shared-download-fixture":
    shared_download()
elif sys.argv[2] == "--download-fixture":
    for mode in ("tui", "tui-cancel"):
        with tempfile.TemporaryDirectory(prefix="zg-tui-download-") as tmp:
            root = Path(tmp)
            os.environ["ZG_MONITOR_FIXTURE"] = str(root / "owned-cache")
            os.environ["ZG_MONITOR_MODE"] = mode
            with Terminal(root, ["--ignored", "--exact", "pull::monitor::tests::monitor_child", "--nocapture"], width=150, height=40) as terminal:
                terminal.check_started()
                terminal.send(alt(b"5"), b"Paused", 15)
                menu_action(terminal,0,2,b"Running",15,opener=click(terminal.width-4,1))
                # The existing fixture body accepts 128 bytes per 40 ms.
                # Allow a prefix before asking for the safe range checkpoint;
                # the child asserts that an actual second request was made.
                time.sleep(0.35)
                menu_action(terminal,0,2,b"Pause",15,opener=click(terminal.width-4,1))
                # Ratatui may emit only the final changed cell of "Paused".
                # Change views after the safe body boundary to read a newly
                # painted status, not the raw differential transition bytes.
                time.sleep(0.5)
                terminal.send(alt(b"1"), b"Quick", 15)
                terminal.send(alt(b"5"), b"Paused", 15)
                if mode == "tui":
                    menu_action(terminal,0,2,b"Snapshot",20,opener=click(terminal.width-4,1))
                else:
                    menu_action(terminal,1,2,b"Cancel",20,opener=click(terminal.width-4,1))
                    time.sleep(0.5)
                    terminal.send(alt(b"1"), b"Quick", 15)
                    terminal.send(alt(b"5"), b"Cancelled", 15)
                os.write(terminal.master, b"\x11")
                try:
                    terminal.finish()
                except AssertionError:
                    while select.select([terminal.master], [], [], 0)[0]:
                        terminal.transcript.extend(os.read(terminal.master, 65536))
                    raise AssertionError(f"owned {mode} fixture exit: {terminal.transcript[-4000:]!r}")
                assert b"NOT_IN_OUTPUT" not in terminal.transcript
    print("Native download PTY: actual safe pause, resume/durable and cancel outcomes over one shared owner")
else:
    lifecycle(Path(sys.argv[2]), sys.argv[3])
