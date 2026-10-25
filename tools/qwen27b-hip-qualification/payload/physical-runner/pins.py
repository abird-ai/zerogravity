"""Held regular-file identities; exact closure checks before and after execution."""
import hashlib
import json
import os
from pathlib import Path
import stat
import signal
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def alarm_until(deadline):
    """One absolute local phase bound; retries never reset its allowance."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('local supervisor phase deadline')
    signal.setitimer(signal.ITIMER_REAL, remaining)


def stamp(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


_MANAGED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)


def acquire_fd(path):
    """Open/register one FD atomically against managed signals; hash outside."""
    old = signal.pthread_sigmask(signal.SIG_BLOCK, _MANAGED)
    fd = None
    file = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        file = os.fdopen(fd, 'rb')
        fd = None
        # Unmask while the file object is still held here: a pending managed
        # signal may raise, and the except closes the already-registered object.
        signal.pthread_sigmask(signal.SIG_SETMASK, old)
        old = None
        return file
    except BaseException:
        if fd is not None:
            try:
                os.close(fd)
            except BaseException:
                pass
        if file is not None:
            try:
                file.close()
            except BaseException:
                pass
        raise
    finally:
        if old is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)


class Held:
    def __init__(self, path, expected=None, limit=16 * 1024**3):
        self.path = Path(path)
        self.file = None
        try:
            self.file = acquire_fd(path)
            s = os.fstat(self.file.fileno())
            self.before = stamp(s)
            require(stat.S_ISREG(s.st_mode) and s.st_size <= limit, 'bounded regular file: '+str(path))
            self.proof = {'bytes': s.st_size, 'sha256': hashlib.file_digest(self.file, 'sha256').hexdigest()}
            self.identity()
            require(expected is None or self.proof == expected, 'held file differs: '+str(path))
            self.file.seek(0)
        except BaseException:
            self.close()
            raise

    def identity(self):
        require(stamp(os.fstat(self.file.fileno())) == self.before
                and stamp(self.path.stat(follow_symlinks=False)) == self.before,
                'held/path identity changed: '+str(self.path))

    def after(self):
        self.identity()
        self.file.seek(0)
        require(hashlib.file_digest(self.file, 'sha256').hexdigest() == self.proof['sha256'],
                'held bytes changed: '+str(self.path))
        self.identity()
        self.file.seek(0)

    def close(self):
        if self.file is not None:
            self.file.close()


def pin(path, limit=16*1024**3):
    held = Held(path, limit=limit)
    try:
        return held.proof
    finally:
        held.close()


def parsed(path, limit=1024**2):
    held = Held(path, limit=limit)
    try:
        value = json.load(held.file)
        held.after()
        return value, held.proof
    finally:
        held.close()


def relative(value):
    p = Path(value)
    require(not p.is_absolute() and p.parts and '..' not in p.parts
            and p.as_posix() == value, 'safe canonical relative member')
    return p


def check_closure(base, closure):
    for name, expected in closure['records'].items():
        require(pin(base / relative(name)) == expected, 'closure changed: '+name)


def hold_closure(base, closure):
    held = []
    try:
        for name, expected in sorted(closure['records'].items()):
            held.append(Held(base / relative(name), expected))
        return held
    except BaseException:
        for item in held:
            item.close()
        raise
