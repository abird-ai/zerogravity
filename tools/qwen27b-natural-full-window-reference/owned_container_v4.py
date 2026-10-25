"""ROOT ONLY finite owned lifetime. No execution occurs by importing this file.

Additive v4: original nonce/CID binding and commands preserved. WNOWAIT and
selector cleanup follow the reviewed AV runner, with whole-group KILL strictly
before leader reap: its reserved PID prevents accidental process-group reuse.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time

PODMAN = ["podman", "--root", "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers", "--runroot", "/run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers"]
LABEL = "abird.zerogravity.vllm-gpu-oracle-owner"
CAP = 4 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def command(argv, timeout=10):
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=False)
    require(len(result.stdout) + len(result.stderr) <= 1024 * 1024, "bounded command receipt")
    return result


def leader_status_unreaped(process):
    require(process.returncode is None, "owned leader already reaped; never signal a recycled PGID")
    observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if observed is None:
        return None
    require(observed.si_pid == process.pid and observed.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED),
            "owned leader exit observation")
    return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status


class Log:
    """Main-thread bounded drain; no thread/future can survive cleanup."""
    def __init__(self, pipe, path):
        self.pipe, self.path = pipe, Path(path)
        self.selector = self.file = None
        self.retained = self.total = 0
        self.eof = self.closed = False
        self.errors = []
        try:
            self.file = self.path.open("xb")
            os.chmod(self.path, 0o600)
            os.set_blocking(pipe.fileno(), False)
            self.selector = selectors.DefaultSelector()
            self.selector.register(pipe, selectors.EVENT_READ)
        except BaseException:
            if self.selector is not None:
                self.selector.close()
            if self.file is not None:
                self.file.close()
            raise

    def drain(self, timeout):
        for key, _ in self.selector.select(timeout):
            # One bounded block per wake; the exit/deadline remains observable
            # even when an owned writer continuously produces output.
            try:
                raw = os.read(key.fd, 65536)
            except BlockingIOError:
                continue
            if not raw:
                self.selector.unregister(key.fileobj)
                self.eof = True
                continue
            self.total += len(raw)
            keep = raw[:max(0, CAP - self.retained)]
            if keep:
                try:
                    self.file.write(keep)
                    self.retained += len(keep)
                except OSError as error:
                    self.errors.append(repr(error))

    def observe(self, process, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.drain(0.05)
            require(self.total <= CAP and not self.errors, "owned log failed or exceeded4MiB")
            status = leader_status_unreaped(process)
            if status is not None:
                return status
        raise TimeoutError("owned foreground exit deadline (leader remains unreaped)")

    def finish(self):
        failures = []
        try:
            deadline = time.monotonic() + 3
            while not self.eof and time.monotonic() < deadline:
                self.drain(0.05)
            require(self.eof, "owned stdout pipe EOF deadline")
            require(self.total <= CAP and not self.errors, "owned log evidence failed/truncated")
            self.file.flush()
            os.fsync(self.file.fileno())
        except BaseException as error:
            failures.append("drain/evidence: " + repr(error))
        finally:
            # Close all three even when EOF, evidence I/O or one close fails.
            for name, close in (("selector", self.selector.close), ("log", self.file.close),
                                ("pipe", self.pipe.close)):
                try:
                    close()
                except BaseException as error:
                    failures.append(name + ": " + repr(error))
            self.closed = self.pipe.closed and self.file.closed and not self.selector.get_map()
        if failures:
            raise RuntimeError("owned log cleanup: " + "; ".join(failures))


def owned_container(name, nonce, cidfile):
    target, recorded_cid = name, None
    if cidfile.exists():
        cid = cidfile.read_text().strip()
        if cid:
            require(re.fullmatch(r"[a-f0-9]{64}", cid), "invalid owned CID; no container signal")
            target = recorded_cid = cid
    exists = command(PODMAN + ["container", "exists", target])
    if exists.returncode == 1:
        return None
    require(exists.returncode == 0, "owned container existence check failed")
    inspected = command(PODMAN + ["container", "inspect", target])
    require(inspected.returncode == 0, "owned container inspect failed")
    value, = json.loads(inspected.stdout)
    require(value["Name"] == name and value["Config"].get("Labels", {}).get(LABEL) == nonce,
            "container name/nonce mismatch; no signal")
    cid = value["Id"]
    require(re.fullmatch(r"[a-f0-9]{64}", cid), "inspected owned CID invalid")
    if recorded_cid is not None:
        require(recorded_cid == cid, "owned CID changed; no signal")
    return cid


def cleanup_container(name, nonce, cidfile):
    errors = []
    try:
        cid = owned_container(name, nonce, cidfile)
    except BaseException as error:
        return ["container owner binding: " + repr(error)]
    if cid is None:
        return errors
    try:
        stopped = command(PODMAN + ["stop", "--time", "5", cid], 15)
        if stopped.returncode != 0:
            errors.append("owned stop returned nonzero")
    except BaseException as error:
        errors.append("container stop: " + repr(error))
    try:
        # Fresh exact owner binding is required even after a failed/timedout
        # stop. A stop failure must not skip the bounded force-remove attempt.
        if owned_container(name, nonce, cidfile) is not None:
            removed = command(PODMAN + ["rm", "--force", cid], 15)
            require(removed.returncode == 0, "owned container force cleanup failed")
    except BaseException as error:
        errors.append("container force cleanup: " + repr(error))
    try:
        require(owned_container(name, nonce, cidfile) is None, "owned container survived cleanup")
    except BaseException as error:
        errors.append("container final absence: " + repr(error))
    return errors


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def signal_group_unreaped(process, sig):
    # waitid authenticates that this is still our unreaped child. start_new_session
    # made it the session/group leader; no foreign group is inferred from a PID.
    leader_status_unreaped(process)
    require(os.getpgid(process.pid) == process.pid and os.getsid(process.pid) == process.pid,
            "owned leader session/PGID binding")
    os.killpg(process.pid, sig)


def cleanup_group(process, receipt, *, observe=leader_status_unreaped, pause=time.sleep):
    """Independent bounded stages; no nonzero signal after any reap attempt."""
    state = receipt["foreground_cleanup"] = {"pgid": process.pid, "term_sent": False,
        "kill_sent_before_reap": False, "leader_reaped": False, "group_absent": False,
        "kill_attempts_before_reap": 0, "leader_reap_attempts": 0}
    errors = []
    try:
        signal_group_unreaped(process, signal.SIGTERM)
        state["term_sent"] = True
    except ProcessLookupError:
        state["term_group_absent_before_reap"] = True
    except BaseException as error:
        errors.append("owned TERM: " + repr(error))
    try:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            observe(process)
            pause(0.05)
    except BaseException as error:
        errors.append("owned grace: " + repr(error))
    # Both independently authenticated KILL attempts precede ANY wait/reap.
    for attempt in range(2):
        state["kill_attempts_before_reap"] += 1
        try:
            signal_group_unreaped(process, signal.SIGKILL)
            state["kill_sent_before_reap"] = True
        except ProcessLookupError:
            state["kill_group_absent_before_reap"] = True
        except BaseException as error:
            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
    state["all_nonzero_signals_before_reap"] = True
    # Popen.wait targets only this child. A failed first wait cannot skip final wait.
    try:
        state["leader_reap_attempts"] += 1
        state["exit_code"] = process.wait(timeout=5)
        state["leader_reaped"] = True
    except BaseException as error:
        errors.append("owned first leader reap: " + repr(error))
    finally:
        if process.returncode is None:
            try:
                state["leader_reap_attempts"] += 1
                state["exit_code"] = process.wait(timeout=5)
                state["leader_reaped"] = True
            except BaseException as error:
                errors.append("owned final leader reap: " + repr(error))
        else:
            state["exit_code"] = process.returncode
            state["leader_reaped"] = True
    try:
        require(state["leader_reaped"], "owned leader was not reaped")
        deadline = time.monotonic() + 5
        while group_exists(process.pid) and time.monotonic() < deadline:
            pause(0.05)  # group_exists uses signal0 only; never signal after reap.
        state["group_absent"] = not group_exists(process.pid)
        require(state["group_absent"], "owned foreground group remains after final KILL/reap")
    except BaseException as error:
        errors.append("owned group absence: " + repr(error))
    state["stage_errors"] = list(errors)
    if errors:
        raise RuntimeError("independent owned foreground cleanup: " + "; ".join(errors))


def cleanup(process, name, nonce, cidfile, receipt):
    errors = []
    try:
        errors.extend(cleanup_container(name, nonce, cidfile))
    except BaseException as error:
        errors.append("initial owned container cleanup: " + repr(error))
    try:
        cleanup_group(process, receipt)
    except BaseException as error:
        errors.append("owned foreground cleanup: " + repr(error))
    # Rediscover startup races using only the original nonce/name/CID binding.
    try:
        errors.extend(cleanup_container(name, nonce, cidfile))
    except BaseException as error:
        errors.append("final owned container cleanup: " + repr(error))
    try:
        require(receipt.get("foreground_cleanup", {}).get("leader_reaped"), "owned leader not reaped")
        require(receipt["foreground_cleanup"]["group_absent"], "owned group absence not established")
        require(owned_container(name, nonce, cidfile) is None, "owned container appeared after foreground reap")
        receipt["owned_container_absent_after_reap"] = True
    except BaseException as error:
        errors.append("final owned absence: " + repr(error))
    return errors
