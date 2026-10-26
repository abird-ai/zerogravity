#!/usr/bin/env python3
"""Owned foreground provider processes for serialized serving comparison.

Lifecycle receipts establish process observations, not source/math/GPU parity.
No shell, existing-process discovery, driver changes, or cache eviction.

Strict spec: schema, policy, startup_timeout_seconds, shutdown_timeout_seconds,
log_limit_bytes, providers {zerogravity, vllm}. Each provider requires argv and
declarations {runtime, source_sha256, tokenizer_sha256, numerical_profile,
quantization, generation_policy, hardware, resource_limits}; optional cwd, env,
redact_arg_indices and stop_argv. Policies both spawn a fresh process per sample;
warmed_process also runs one unmeasured request. cold_process evicts no caches.
Podman must run foreground with --cidfile {cidfile}; its stop argv must share
the launch prefix then use stop --ignore --time N {owned_cid}. An optional
{owned_name} placeholder expands to a fresh generated name. Missing CID or a
failed stop blocks the next sample. Receipts retain environment names only;
raw child output is private and capped, then drained without retention.
Stop stdout/stderr must be fully retained within each stream's cap; loss,
missing EOF, timeout or escalation rejects the next sample. Exact-CID container
State observations are bounded and optional; auto-removal can make terminal
state unavailable. Exit 137 and absent inspect data never establish OOM.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import re
import selectors
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid


PROVIDERS = {"zerogravity", "vllm"}
POLICIES = {"cold_process", "warmed_process"}
SCHEMA = "zerogravity.serving.provider-lifecycle.v1"


def finite(value, low, high):
    return type(value) in {int, float} and math.isfinite(value) and low <= value <= high


def argv(value):
    if (not isinstance(value, list) or not 1 <= len(value) <= 512
            or any(not isinstance(v, str) or not v or "\0" in v or len(v.encode()) > 16384 for v in value)
            or sum(len(v.encode()) for v in value) > 128 * 1024):
        raise ValueError("lifecycle argv must be bounded nonempty direct arguments")
    if Path(value[0]).name in {"sh", "bash", "dash", "zsh", "ksh", "fish", "cmd", "powershell"}:
        raise ValueError("lifecycle commands must execute foreground programs directly, without a shell")
    return list(value)


def endpoint(url):
    parsed = urllib.parse.urlsplit(url)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError as error:
        raise ValueError("lifecycle endpoint needs a literal loopback HTTP address") from error
    if (parsed.scheme != "http" or not address.is_loopback or port is None or not 1 <= port <= 65535
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("lifecycle endpoint needs literal loopback HTTP, a port and no credentials/path/query")
    return str(address), port


def listening(address):
    try:
        with socket.create_connection(address, timeout=.15):
            return True
    except ConnectionRefusedError:
        return False
    except OSError as error:
        # A timeout or other unknown transport state is not evidence of vacancy.
        raise ValueError("cannot establish unused lifecycle endpoint") from error


def validate(spec):
    required = {"schema", "policy", "startup_timeout_seconds", "shutdown_timeout_seconds", "log_limit_bytes", "providers"}
    if (not isinstance(spec, dict) or set(spec) != required or spec.get("schema") != SCHEMA
            or not isinstance(spec.get("policy"), str) or spec["policy"] not in POLICIES
            or not finite(spec.get("startup_timeout_seconds"), 1, 600)
            or not finite(spec.get("shutdown_timeout_seconds"), 1, 60)
            or type(spec.get("log_limit_bytes")) is not int or not 1024 <= spec["log_limit_bytes"] <= 8*1024*1024
            or not isinstance(spec.get("providers"), dict) or set(spec["providers"]) != PROVIDERS):
        raise ValueError("invalid strict bounded provider lifecycle envelope")
    declarations = {"runtime", "source_sha256", "tokenizer_sha256", "numerical_profile", "quantization", "generation_policy", "hardware", "resource_limits"}
    for provider in spec["providers"].values():
        if (not isinstance(provider, dict) or not {"argv", "declarations"} <= set(provider)
                or set(provider) - {"argv", "declarations", "cwd", "env", "redact_arg_indices", "stop_argv"}):
            raise ValueError("invalid strict lifecycle provider")
        command = argv(provider["argv"])
        cwd = provider.get("cwd")
        if cwd is not None and (not isinstance(cwd, str) or not Path(cwd).is_absolute() or not Path(cwd).is_dir()):
            raise ValueError("lifecycle cwd must be an existing absolute directory")
        env = provider.get("env", {})
        if (not isinstance(env, dict) or len(env) > 128
                or any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", k)
                       or k in {"HOME", "home", "CODEX_HOME"} or not isinstance(v, str) or "\0" in v or len(v.encode()) > 16384
                       for k, v in env.items()) or sum(len(k)+len(v) for k,v in env.items()) > 128*1024):
            raise ValueError("invalid bounded lifecycle environment overrides")
        redacted = provider.get("redact_arg_indices", [])
        if (not isinstance(redacted, list) or any(type(i) is not int or not 0 <= i < len(command) for i in redacted)
                or len(set(redacted)) != len(redacted)):
            raise ValueError("invalid lifecycle redacted argv indices")
        declaration = provider["declarations"]
        if (not isinstance(declaration, dict) or set(declaration) != declarations
                or any(not isinstance(declaration.get(k), str) or not declaration[k] or len(declaration[k].encode()) > 4096 for k in declarations - {"resource_limits"})
                or any(not re.fullmatch(r"[a-f0-9]{64}", declaration[k]) for k in ["source_sha256", "tokenizer_sha256"])
                or not isinstance(declaration["resource_limits"], dict) or not declaration["resource_limits"]
                or len(json.dumps(declaration, allow_nan=False).encode()) > 32*1024):
            raise ValueError("lifecycle requires bounded caller runtime/source/tokenizer/profile/policy/hardware/cap declarations")
        if "{owned_cid}" in "\0".join(command):
            raise ValueError("owned CID is available only to a verified stop command")
        stop = provider.get("stop_argv")
        if Path(command[0]).name == "podman" and stop is None:
            raise ValueError("podman lifecycle requires an owned CID stop command")
        if stop is not None:
            stop = argv(stop)
            try:
                run = command.index("run")
            except ValueError as error:
                raise ValueError("owned container stop requires direct foreground podman run") from error
            suffix = stop[run:]
            if (Path(command[0]).name != "podman" or stop[:run] != command[:run]
                    or suffix[:3] != ["stop", "--ignore", "--time"] or len(suffix) != 5
                    or not re.fullmatch(r"[0-9]{1,2}", suffix[3]) or int(suffix[3]) > spec["shutdown_timeout_seconds"]
                    or suffix[4] != "{owned_cid}"
                    or "--cidfile" not in command or command[command.index("--cidfile")+1:command.index("--cidfile")+2] != ["{cidfile}"]
                    or command.count("{cidfile}") != 1 or command.count("--cidfile") != 1
                    or any(v in {"-d", "--detach", "--replace", "--remote", "--url", "--connection"}
                           or v.startswith(("--detach=", "--replace=", "--url=", "--connection=", "--remote=")) for v in command)
                    or any(v in {"-c", "-H"} or v.startswith(("-c=", "-H=")) for v in command[1:run])):
                raise ValueError("stop_argv must target only fresh --cidfile via same podman prefix: stop --ignore --time N {owned_cid}")
    return spec


def redacted_argv(command, indices=()):
    result = list(command)
    hidden = set(indices)
    secret_flags = {"--api-key", "--token", "--password", "--hf-token", "--access-token"}
    for i, value in enumerate(command):
        if value in secret_flags and i+1 < len(command):
            hidden.add(i+1)
        if any(value.startswith(flag+"=") for flag in secret_flags):
            result[i] = value.split("=", 1)[0] + "=[REDACTED]"
    for index in hidden:
        result[index] = "[REDACTED]"
    return result


def write_json(path, value):
    temporary = path.with_suffix(".writing")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def stamp():
    return {"utc":datetime.now(timezone.utc).isoformat(), "monotonic_ns":time.monotonic_ns()}


def leader_status_unreaped(process):
    if process.returncode is not None:
        raise ChildProcessError("owned leader already reaped; cannot authenticate group signaling")
    observed = os.waitid(os.P_PID,process.pid,os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if observed is None:
        return None
    if observed.si_pid != process.pid or observed.si_code not in {os.CLD_EXITED,os.CLD_KILLED,os.CLD_DUMPED}:
        raise ChildProcessError("unexpected owned child exit observation")
    return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status


def terminal_outcome(code):
    # Positive 128+signal exit statuses do not prove a signal, much less OOM.
    return {"kind":"unknown" if code is None else "signal" if code < 0 else "exit",
            "returncode":code, "signal":-code if code is not None and code < 0 else None,
            "oom":"not_established"}


def owned_group_members(process):
    """Only the original pinned session/group; zombies are not live workers."""
    leader_status_unreaped(process)
    if os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid:
        raise ChildProcessError("owned child lost its original session/group binding")
    live, zombies = [], []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry/"stat").read_text().rsplit(") ",1)[1].split()
        except (FileNotFoundError,ProcessLookupError):
            continue
        if int(fields[2]) == process.pid:
            if int(fields[3]) != process.pid:
                raise ChildProcessError("owned group session mismatch")
            (zombies if fields[0] in {"Z","X"} else live).append(int(entry.name))
    return {"live_pids":sorted(live), "zombie_pids":sorted(zombies)}


def retire_child(process, record, natural_deadline=None):
    """All nonzero group signals precede the one original-child reap attempt."""
    for key,value in {"signals":[],"leader_reaped":False,"leader_reap_attempts":0,"errors":[],"timed_out":False}.items():
        record.setdefault(key,value)
    record.setdefault("retirement_attempts",[]).append(stamp())
    if record["leader_reaped"] or record["leader_reap_attempts"]:
        record["errors"].append({"stage":"recovery_after_reap_attempt","error_type":"ChildProcessError"})
        record["complete"] = False
        return False
    def observe():
        try:
            code = leader_status_unreaped(process)
            members = owned_group_members(process)
            record["last_group_observation"] = {**stamp(),**members}
            return code,members["live_pids"]
        except Exception as error:
            record["errors"].append({"stage":"observe_unreaped","error_type":type(error).__name__})
            return None,None
    code,live = observe()
    if natural_deadline is not None:
        while code is None and live is not None and time.monotonic() < natural_deadline:
            time.sleep(min(.01,max(0,natural_deadline-time.monotonic())))
            code,live = observe()
        record["timed_out"] |= code is None and time.monotonic() >= natural_deadline
    if "terminal_before_cleanup" in record:
        record.setdefault("recovery_observations",[]).append({**stamp(),"terminal_outcome":terminal_outcome(code)})
    else:
        record["terminal_before_cleanup"] = terminal_outcome(code)
    for sig,grace in [(signal.SIGTERM,.3),(signal.SIGKILL,1.)]:
        if live == []:
            break
        attempt = {"signal":sig.name,**stamp(),"sent":False,"leader_unreaped":False}
        record["signals"].append(attempt)
        try:
            leader_status_unreaped(process)
            if os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid:
                raise ChildProcessError("owned child lost its original session/group binding")
            attempt["leader_unreaped"] = True
            # A call interrupted after dispatch cannot be reported as unsent.
            attempt["sent"] = None
            os.killpg(process.pid,sig)
            attempt["sent"] = True
        except ProcessLookupError:
            attempt["sent"] = False
            attempt["group_absent"] = True
        except Exception as error:
            attempt["error_type"] = type(error).__name__
            record["errors"].append({"stage":"signal","error_type":type(error).__name__})
        until = time.monotonic()+grace
        while time.monotonic() < until:
            code,live = observe()
            if code is not None and live == []:
                break
            time.sleep(.01)
    record["all_nonzero_signals_before_reap"] = True
    record["terminal_before_reap"] = terminal_outcome(code)
    record["returncode"] = None
    record["reap_start"] = stamp()
    try:
        # Popen.wait invents status 0 on ECHILD; authenticate before using it.
        leader_status_unreaped(process)
        record["leader_reap_attempts"] = 1
        record["returncode"] = process.wait(timeout=1)
        record["leader_reaped"] = True
    except Exception as error:
        record["errors"].append({"stage":"reap","error_type":type(error).__name__})
    record["reap_end"] = stamp()
    # Observation only after reap: the numeric group is never signaled again.
    until = time.monotonic()+.5
    try:
        while group_alive(process.pid) and time.monotonic() < until:
            time.sleep(.01)
        record["owned_process_group_gone"] = not group_alive(process.pid)
    except Exception as error:
        record["owned_process_group_gone"] = False
        record["errors"].append({"stage":"group_absence","error_type":type(error).__name__})
    record["terminal_outcome"] = terminal_outcome(record["returncode"])
    record["cleanup_complete"] = bool(record["leader_reaped"] and record["owned_process_group_gone"])
    record["complete"] = record["cleanup_complete"] and not record["errors"]
    return record["complete"]


class BoundedLog:
    def __init__(self, pipe, path, limit):
        self.pipe, self.path, self.limit = pipe, path, limit
        self.seen = self.retained = 0
        self.digest = hashlib.sha256()
        self.error = None
        self.eof = False
        self.eof_at = None
        self.stop = threading.Event()
        os.set_blocking(pipe.fileno(),False)
        self.thread = threading.Thread(target=self.drain, daemon=True)
        self.thread.start()

    def drain(self):
        target = None
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            target = os.fdopen(descriptor, "wb")
        except Exception as error:
            self.error = type(error).__name__
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.pipe, selectors.EVENT_READ)
                while not self.stop.is_set():
                    if not selector.select(.05):
                        continue
                    block = os.read(self.pipe.fileno(),65536)
                    if not block:
                        self.eof = True
                        self.eof_at = stamp()
                        break
                    self.seen += len(block)
                    if target is not None and self.error is None:
                        keep = block[:max(0,self.limit-self.retained)]
                        try:
                            if target.write(keep) != len(keep):
                                raise OSError("short private log write")
                            self.digest.update(keep)
                            self.retained += len(keep)
                        except Exception as error:
                            self.error = type(error).__name__
            if target is not None:
                target.flush()
                os.fsync(target.fileno())
        except Exception as error:
            self.error = type(error).__name__
        finally:
            if target is not None:
                try:
                    target.close()
                except Exception as error:
                    self.error = type(error).__name__

    def finish(self, seconds):
        self.thread.join(max(0, seconds))
        if self.thread.is_alive():
            self.stop.set()
            self.thread.join(.2)
        done = not self.thread.is_alive()
        if done:
            self.pipe.close()
        return {"path":str(self.path), "bytes_seen":self.seen, "bytes_retained":self.retained,
                "truncated":self.seen > self.retained, "sha256_retained":self.digest.hexdigest(),
                "reader_complete":done, "eof":self.eof, "eof_observed":self.eof_at, "reader_error":self.error,
                "capture_complete":done and self.eof and self.error is None and self.seen == self.retained}


def owned_command(command, label, cid, seconds, checkpoint, *, directory,
                  container_name, log_limit_bytes, cwd=None, env=None,
                  redact_arg_indices=()):
    """Run/reap one owned command; persist intent and complete separate streams.

    checkpoint(stage, record) returns whether the current facts were persisted.
    A caller retains its original container/process owner and acceptance policy.
    """
    record = {"status":"intent", "argv":redacted_argv(command,redact_arg_indices),
        "owned_cid":cid, "owned_container_name":container_name, "intent":stamp(),
        "intent_persisted":True, "pid":None, "pgid":None,
        "timeout_seconds":seconds, "stdout":None, "stderr":None,
        "returncode":None, "timed_out":False, "signals":[], "leader_reaped":False}
    if not checkpoint(label+"_intent",record):
        record["intent_persisted"] = False
    process,logs = None,{}
    began = time.monotonic()
    record["start"] = stamp()
    record["start_observation"] = "immediately_before_Popen"
    try:
        process = subprocess.Popen(command,cwd=cwd,env={**os.environ,**(env or {})},
            stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,bufsize=0)
        record.update(status="running",pid=process.pid,pgid=process.pid)
        checkpoint(label+"_spawn",record)
        for stream in ["stdout","stderr"]:
            logs[stream] = BoundedLog(getattr(process,stream),Path(directory)/(label+"."+stream+".log"),log_limit_bytes)
        retire_child(process,record,began+seconds)
        if record["errors"]:
            record.update(status="command_error",error_type=record["errors"][0]["error_type"])
    except BaseException as error:
        record.update(status="program_not_found" if isinstance(error,FileNotFoundError) else "command_error",error_type=type(error).__name__)
        record.setdefault("errors",[]).append({"stage":"command","error_type":type(error).__name__})
        if process is not None and process.returncode is None and not record.get("leader_reap_attempts"):
            retire_child(process,record)
    finally:
        for stream in ["stdout","stderr"]:
            try:
                if stream in logs:
                    record[stream] = logs[stream].finish(.5)
                elif process is not None:
                    getattr(process,stream).close()
            except BaseException as error:
                record[stream] = {"reader_complete":False,"eof":False,"capture_complete":False,
                    "reader_error":type(error).__name__}
        record["end"] = stamp()
        record["elapsed_seconds"] = time.monotonic()-began
        record["cleanup_complete"] = process is None or bool(record.get("cleanup_complete")
            and all(record[stream] and record[stream]["reader_complete"] and record[stream]["eof"] for stream in ["stdout","stderr"]))
        record["evidence_complete"] = bool(record["intent_persisted"] and process is not None
            and record["cleanup_complete"] and all(record[stream] and record[stream]["capture_complete"] for stream in ["stdout","stderr"]))
        if record["status"] == "running":
            record["status"] = "timed_out" if record["timed_out"] else "finished"
        record["result_persisted"] = True
        if not checkpoint(label+"_result",record):
            record["result_persisted"] = record["evidence_complete"] = False
    return record


class LifecyclePlan:
    def __init__(self, spec, endpoints, output_dir=None):
        self.spec = validate(spec)
        self.endpoints = {label:endpoint(url) for label,url in endpoints.items()}
        if set(self.endpoints) != PROVIDERS or len(set(self.endpoints.values())) != 2:
            raise ValueError("lifecycle requires two distinct literal loopback endpoints")
        if output_dir is None:
            state = os.environ.get("XDG_STATE_HOME")
            if state and not Path(state).is_absolute():
                raise ValueError("XDG_STATE_HOME must be absolute")
            output_dir = (Path(state) if state else Path.home()/".local/state") / "abird/zerogravity/serve-compare"
        base = Path(output_dir).resolve()
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory = Path(tempfile.mkdtemp(prefix="owned-compare-", dir=base))
        self.current = None
        self.blocked = False
        self.receipts = []
        self.manifest = {"schema":SCHEMA, "policy":spec["policy"], "receipts":self.receipts,
            "spec_sha256":hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()).hexdigest(),
            "semantics":"one fresh owned foreground process per sample; OS/storage/device caches are not evicted or claimed cold; declarations are caller-provided, not independently verified"}
        self.save()

    def save(self):
        write_json(self.directory/"lifecycle.json", self.manifest)

    @contextmanager
    def sample(self, label, preflight):
        if self.current is not None or self.blocked:
            raise ValueError("previous owned provider cleanup is incomplete")
        instance = OwnedProvider(self, label)
        self.current = instance
        self.receipts.append(str(instance.receipt_path))
        self.save()
        try:
            instance.start(preflight)
            yield instance
        except BaseException as error:
            instance.receipt["failure_type"] = type(error).__name__
            instance.save()
            raise
        finally:
            complete = False
            try:
                complete = instance.close()
            finally:
                self.current = None
                # Lost evidence also forbids starting the next provider.
                self.blocked = not complete
            if not complete:
                raise ValueError(f"owned provider cleanup or shutdown evidence/outcome rejected; inspect {instance.receipt_path}")


class OwnedProvider:
    def __init__(self, plan, label):
        self.plan, self.label = plan, label
        self.spec = plan.spec["providers"][label]
        self.directory = plan.directory/f"{len(plan.receipts):03d}-{label}"
        self.directory.mkdir(mode=0o700)
        self.receipt_path = self.directory/"receipt.json"
        self.cidfile = self.directory/"owned.cid"
        self.name = "zg-compare-" + uuid.uuid4().hex
        self.process = self.log = self.probe = self.owned_cid = None
        self.closed = False
        self.began = time.monotonic()
        self.receipt = {"provider":label, "policy":plan.spec["policy"], "created_utc":datetime.now(timezone.utc).isoformat(),
            "declarations":self.spec["declarations"], "declaration_trust":"caller_provided_not_independently_verified",
            "argv_redacted":redacted_argv(self.spec["argv"],self.spec.get("redact_arg_indices",())),
            "env_override_names":sorted(self.spec.get("env",{})), "environment_inherited":True,
            "inherited_env_names":sorted(os.environ),
            "cwd":self.spec.get("cwd"), "owned_container_name":self.name,
            "warmup":None, "startup":{}, "shutdown":None}
        self.save()

    def save(self):
        write_json(self.receipt_path, self.receipt)

    def expand(self, command, cid=None):
        return [value.replace("{cidfile}", str(self.cidfile)).replace("{owned_name}",self.name).replace("{owned_cid}",cid or "") for value in command]

    def idle(self):
        return not any(listening(address) for address in self.plan.endpoints.values())

    def cid(self):
        if self.owned_cid is not None:
            return self.owned_cid
        descriptor = os.open(self.cidfile, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as source:
            value = source.read(129)
        if not re.fullmatch(rb"[a-f0-9]{64}\n?", value):
            raise ValueError("fresh owned container CID is invalid")
        os.chmod(self.cidfile, 0o600)
        # A later file rewrite must never retarget cleanup to another container.
        self.owned_cid = value.strip().decode("ascii")
        self.receipt["owned_cid"] = self.owned_cid
        return self.owned_cid

    def start(self, preflight):
        self.receipt["startup"] = {"ready":False,"spawned":False,"attempts":0,"last_failures":[]}
        self.save()
        if not self.idle():
            self.receipt["startup"] = {"ready":False,"existing_listener":True,"spawned":False}
            self.save()
            raise ValueError(f"existing lifecycle endpoint listener; no process was stopped; inspect {self.receipt_path}")
        command = self.expand(self.spec["argv"])
        self.process = subprocess.Popen(command, cwd=self.spec.get("cwd"), env={**os.environ,**self.spec.get("env",{})},
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
        self.receipt["pid"] = self.receipt["pgid"] = self.process.pid
        self.receipt["startup"]["spawned"] = True
        self.log = BoundedLog(self.process.stdout,self.directory/"provider.log",self.plan.spec["log_limit_bytes"])
        self.save()
        deadline = self.began+self.plan.spec["startup_timeout_seconds"]
        attempts, failures = 0, deque(maxlen=16)
        while time.monotonic() < deadline:
            if self.log.error:
                raise ValueError("owned provider log reader failed")
            if leader_status_unreaped(self.process) is not None:
                raise ValueError(f"owned provider exited before readiness; inspect {self.receipt_path}")
            results = queue.Queue(maxsize=1)
            def probe():
                try:
                    preflight()
                    results.put(None)
                except Exception as error:
                    results.put(type(error).__name__)
            self.probe = threading.Thread(target=probe,daemon=True)
            self.probe.start()
            attempts += 1
            self.receipt["startup"].update(attempts=attempts,elapsed_seconds=time.monotonic()-self.began)
            self.save()
            try:
                failure = results.get(timeout=max(.001, deadline-time.monotonic()))
            except queue.Empty as error:
                raise ValueError(f"owned provider readiness deadline; inspect {self.receipt_path}") from error
            if failure is None:
                other = next(address for label,address in self.plan.endpoints.items() if label != self.label)
                if leader_status_unreaped(self.process) is not None or listening(other):
                    raise ValueError("readiness lost foreground ownership or another provider endpoint appeared")
                if self.spec.get("stop_argv"):
                    self.receipt["owned_cid"] = self.cid()
                self.receipt["startup"].update(ready=True,attempts=attempts,last_failures=list(failures),elapsed_seconds=time.monotonic()-self.began)
                self.save()
                return
            failures.append({"elapsed_seconds":time.monotonic()-self.began,"error_type":failure})
            self.receipt["startup"].update(ready=False,attempts=attempts,last_failures=list(failures),elapsed_seconds=time.monotonic()-self.began)
            self.save()
            time.sleep(min(.1,max(0,deadline-time.monotonic())))
        raise ValueError(f"owned provider readiness deadline; inspect {self.receipt_path}")

    def record_warmup(self, outcome):
        self.receipt["warmup"] = outcome
        self.save()
        self.require_foreground()

    def require_foreground(self):
        other = next(address for label,address in self.plan.endpoints.items() if label != self.label)
        if self.process is None or leader_status_unreaped(self.process) is not None or listening(other):
            raise ValueError(f"lost owned foreground process or another provider appeared; inspect {self.receipt_path}")

    def record_measurement(self, result):
        path = self.directory/"measurement.json"
        write_json(path,result)
        self.receipt["measurement"] = {"path":str(path),"run_id":result["run_id"]}
        self.save()
        self.require_foreground()
        self.receipt["measurement"]["foreground_alive_after_measurement"] = True
        self.save()

    def command(self, command, label, cid, seconds, checkpoint):
        run = self.spec["argv"].index("run")
        redactions = [i for i in self.spec.get("redact_arg_indices",()) if i < run]
        installed = False
        def persist(stage, record):
            nonlocal installed
            if not installed:
                shutdown = self.receipt["shutdown"]
                if label == "stop":
                    shutdown["stop"] = record
                else:
                    shutdown["container_state"].append(record)
                    record["stage"] = label
                installed = True
            return checkpoint(stage)
        return owned_command(command,label,cid,seconds,persist,
            directory=self.directory,container_name=self.name,
            log_limit_bytes=self.plan.spec["log_limit_bytes"],cwd=self.spec.get("cwd"),
            env=self.spec.get("env",{}),redact_arg_indices=redactions)

    def inspect_container(self, cid, stage, checkpoint):
        run = self.spec["argv"].index("run")
        command = self.expand(self.spec["argv"][:run])+["container","inspect","--format","{{json .State}}",cid]
        record = self.command(command,"inspect_"+stage,cid,.25,checkpoint)
        record["state"] = None
        if record.get("returncode") != 0 or not record["evidence_complete"] or record.get("timed_out"):
            record["observation"] = "unavailable"
        else:
            try:
                state = json.loads(Path(record["stdout"]["path"]).read_bytes())
                if (not isinstance(state,dict) or type(state.get("Running")) is not bool
                        or type(state.get("OOMKilled")) is not bool or type(state.get("ExitCode")) is not int
                        or "Error" in state and not isinstance(state["Error"],str)):
                    raise ValueError("container State needs typed Running/OOMKilled/ExitCode observations")
                # Retain State only, never configuration, environment or image data.
                record["state"] = state
                record["observation"] = "observed"
            except (OSError,ValueError,UnicodeError,RecursionError) as error:
                record.update(observation="unavailable",parse_error_type=type(error).__name__)
        checkpoint("inspect_"+stage+"_observation")
        return record

    def close(self):
        if self.closed:
            return self.receipt["shutdown"].get("accepted",False)
        self.closed = True
        began = time.monotonic()
        shutdown = {"signals":[],"stop":None,"container_state":[],"start":stamp(),"complete":False,
            "cleanup_complete":False,"graceful_terminal_outcome":False,"accepted":False,
            "container_oom":"not_established"}
        self.receipt["shutdown"] = shutdown
        evidence_errors = []
        def checkpoint(stage):
            try:
                self.save()
                return True
            except BaseException as error:
                evidence_errors.append(error)
                shutdown["accepted"] = False
                shutdown.setdefault("evidence_write_errors",[]).append({"stage":stage,"error_type":type(error).__name__})
                shutdown["evidence_write_error"] = type(error).__name__
                return False
        if not self.receipt["startup"].get("ready"):
            self.receipt["startup"]["elapsed_seconds"] = time.monotonic()-self.began
        def inspect(cid,stage):
            try:
                self.inspect_container(cid,stage,checkpoint)
            except BaseException as error:
                # Optional observability cannot skip the exact-CID stop attempt.
                shutdown["container_state"].append({"stage":"inspect_"+stage,"owned_cid":cid,
                    "observation":"unavailable","error_type":type(error).__name__,"observed":stamp()})
        stop_ok = True
        if self.process is not None and self.spec.get("stop_argv"):
            try:
                cid = self.cid()
                inspect(cid,"before_stop")
                command = self.expand(self.spec["stop_argv"],cid)
                stop = self.command(command,"stop",cid,self.plan.spec["shutdown_timeout_seconds"],checkpoint)
                stop_ok = bool(stop.get("status") == "finished" and not stop.get("error_type") and stop.get("returncode") == 0
                    and not stop.get("timed_out") and not stop.get("signals") and stop["evidence_complete"])
                inspect(cid,"after_stop")
            except FileNotFoundError:
                stop_ok = False
                shutdown["stop"] = {"status":"no_fresh_CID" if self.owned_cid is None else "stop_program_not_found",
                    "foreground_exited":leader_status_unreaped(self.process) is not None}
            except BaseException as error:
                stop_ok = False
                if shutdown["stop"] is None:
                    shutdown["stop"] = {"error_type":type(error).__name__}
                else:
                    shutdown["stop"]["cleanup_error_type"] = type(error).__name__
        if self.process is not None:
            foreground = shutdown["foreground"] = {}
            # A successful stop may precede its foreground attach process exit.
            retire_child(self.process,foreground,time.monotonic()+.25 if self.spec.get("stop_argv") else None)
            shutdown["signals"] = [item["signal"] for item in foreground["signals"] if item["sent"] is not False]
            shutdown["returncode"] = foreground["returncode"]
            shutdown["owned_process_group_gone"] = foreground["owned_process_group_gone"]
            shutdown["provider_terminal_outcome"] = terminal_outcome(foreground["returncode"])
        else:
            shutdown["owned_process_group_gone"] = True
            shutdown["provider_terminal_outcome"] = terminal_outcome(None)
        if self.probe:
            self.probe.join(.5)
        shutdown["readiness_reader_complete"] = self.probe is None or not self.probe.is_alive()
        if self.log:
            self.receipt["log"] = self.log.finish(.5)
        elif self.process is not None and self.process.stdout is not None:
            self.process.stdout.close()
        try:
            shutdown["endpoints_closed"] = self.idle()
        except Exception as error:
            shutdown["endpoints_closed"] = False
            shutdown["endpoint_observation_error"] = type(error).__name__
        shutdown["cleanup_complete"] = self.process is None or bool(shutdown["foreground"]["cleanup_complete"] and shutdown["readiness_reader_complete"]
            and (not self.log or self.receipt["log"]["reader_complete"] and self.receipt["log"]["eof"] and not self.receipt["log"]["reader_error"])
            and shutdown["endpoints_closed"])
        commands = shutdown["container_state"]+([shutdown["stop"]] if shutdown["stop"] else [])
        shutdown["cleanup_complete"] &= all(item.get("cleanup_complete",False) for item in commands if item.get("pid") is not None)
        shutdown["complete"] = shutdown["cleanup_complete"] and stop_ok
        terminal_state = next((item["state"] for item in reversed(shutdown["container_state"])
            if item.get("observation") == "observed" and item["state"].get("Running") is False),None)
        container_ok = not any(item.get("parse_error_type") for item in shutdown["container_state"])
        if terminal_state is not None:
            shutdown["container_oom"] = "observed_true" if terminal_state.get("OOMKilled") is True else "observed_false" if terminal_state.get("OOMKilled") is False else "not_established"
            container_ok &= type(terminal_state.get("ExitCode")) is int and terminal_state["ExitCode"] == 0 and terminal_state.get("OOMKilled") is False and not terminal_state.get("Error")
        after_stop = next((item["state"] for item in reversed(shutdown["container_state"])
            if item.get("stage") == "inspect_after_stop" and item.get("observation") == "observed"),None)
        if after_stop is not None and after_stop.get("Running") is not False:
            container_ok = False
        shutdown["container_terminal_state_observed"] = terminal_state is not None
        shutdown["graceful_terminal_outcome"] = bool(self.process is not None and shutdown["complete"] and shutdown["returncode"] == 0
            and "SIGKILL" not in shutdown["signals"] and not shutdown["foreground"]["timed_out"]
            and not shutdown["foreground"]["errors"] and container_ok)
        shutdown["accepted"] = bool(shutdown["complete"] and (not self.receipt["startup"].get("ready")
            or self.receipt.get("failure_type") or shutdown["graceful_terminal_outcome"]) and not evidence_errors)
        shutdown["elapsed_seconds"] = time.monotonic()-began
        shutdown["end"] = stamp()
        # Evidence I/O follows physical cleanup; failure cannot skip teardown.
        checkpoint("shutdown_result")
        try:
            self.plan.save()
        except BaseException as error:
            evidence_errors.append(error)
            shutdown["evidence_write_error"] = type(error).__name__
        if evidence_errors:
            shutdown["accepted"] = False
            checkpoint("shutdown_evidence_failure")
            raise evidence_errors[0]
        return shutdown["accepted"]


def self_test(comparison):
    """Real harmless owned localhost children; no GPU/container/model execution."""
    import copy
    import sys
    fixture_source = r'''
import fcntl,json,os,signal,sys,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
label,port,events,lock,mode=sys.argv[1:6]
lease=open(lock,"w");fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
def event(kind,**fields):
 fd=os.open(events,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
 with os.fdopen(fd,"a") as out:out.write(json.dumps(dict(kind=kind,provider=label,pid=os.getpid(),**fields))+"\n")
event("start");print("x"*8192,flush=True)
if mode=="exit":sys.exit(7)
if mode=="ignore_term":signal.signal(signal.SIGTERM,signal.SIG_IGN)
else:signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
class Provider(BaseHTTPRequestHandler):
 def do_GET(self):
  if mode=="hang":time.sleep(30)
  data=json.dumps(dict(data=[dict(id="wrong" if mode=="wrong" else "fixture")])).encode()
  self.send_response(200);self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers["Content-Length"])));wire=self.headers.get("X-Request-ID")
  event("request",wire=wire);self.send_response(200);self.send_header("Content-Type","text/event-stream");self.end_headers()
  if mode=="warmup_error":docs=[dict(error=dict(code="fixture_error"))]
  else:
   docs=[dict(choices=[dict(index=0,delta=dict(content="ok"),finish_reason=None)]),dict(choices=[dict(index=0,delta={},finish_reason="length")]),dict(choices=[],usage=dict(completion_tokens=1))]
   for doc in docs:doc["id"]="chatcmpl-"+wire
   if body.get("return_token_ids"):docs[0]["choices"][0]["token_ids"]=[10]
  for doc in docs:self.wfile.write(b"data: "+json.dumps(doc).encode()+b"\n\n")
  self.wfile.write(b"data: [DONE]\n\n")
 def log_message(self,*_):pass
server=ThreadingHTTPServer(("127.0.0.1",int(port)),Provider)
try:server.serve_forever()
finally:server.server_close();event("stop");lease.close()
'''
    def vacant():
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1",0))
            return reserved.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="zg-owned-provider-fixture-") as temp:
        root = Path(temp)
        script = root/"provider.py"
        script.write_text(fixture_source)
        events, lock = root/"events.jsonl", root/"exclusive.lock"
        ports = [vacant(),vacant()]
        while ports[0] == ports[1]:
            ports[1] = vacant()
        urls = [f"http://127.0.0.1:{port}" for port in ports]
        def spec(policy="cold_process", mode="normal"):
            return dict(schema=SCHEMA,policy=policy,startup_timeout_seconds=1,shutdown_timeout_seconds=2,log_limit_bytes=1024,
                providers={label:dict(argv=[sys.executable,"-u",str(script),label,str(port),str(events),str(lock),mode,"--api-key","secret-argument"],
                    env={"LIFECYCLE_FIXTURE_SECRET":"secret-environment"},declarations=dict(runtime="synthetic-python-provider",source_sha256="a"*64,
                        tokenizer_sha256="b"*64,numerical_profile="source-free-fixture",quantization="none",generation_policy="one harmless token",
                        hardware="CPU localhost fixture",resource_limits={"processes":1,"gpu":False})) for label,port in zip(["zerogravity","vllm"],ports)})
        def run(settings, repetitions=1):
            return comparison(*urls,"fixture","fixture","Hi.",1,[1],1,None,None,("a"*64,"a"*64),("b"*64,"b"*64),repetitions,3,
                provider_lifecycle=settings,lifecycle_output_dir=str(root/"receipts"),return_token_ids=True)
        result = run(spec(),2)
        expected = ["zerogravity","vllm","vllm","zerogravity","vllm","zerogravity","zerogravity","vllm"]
        actual = [json.loads(line) for line in events.read_text().splitlines()]
        assert [event["provider"] for event in actual if event["kind"] == "start"] == expected
        assert [event["kind"] for event in actual] == [kind for _ in expected for kind in ["start","request","stop"]]
        assert len({event["pid"] for event in actual if event["kind"] == "start"}) == 8
        assert result["warmups_per_endpoint"] == 0
        assert all(row["vllm_equivalent"] == "NOT_ESTABLISHED" for row in result["summaries"])
        for row in result["runs"]:
            receipt = row["provider_lifecycle_receipt"]
            assert receipt["warmup"] is None and receipt["shutdown"]["complete"]
            assert receipt["log"]["truncated"] and receipt["log"]["bytes_retained"] == 1024
            raw = Path(row["provider_lifecycle_receipt_path"]).read_text()
            assert "secret-argument" not in raw and "secret-environment" not in raw
            for path in [Path(row["provider_lifecycle_receipt_path"]),Path(receipt["log"]["path"]),Path(receipt["measurement"]["path"])]:
                assert path.stat().st_mode & 0o777 == 0o600
            assert not group_alive(receipt["pgid"])
        assert not any(listening(endpoint(url)) for url in urls)
        prior = len(actual)
        warmed = run(spec("warmed_process"))
        actual = [json.loads(line) for line in events.read_text().splitlines()][prior:]
        assert [event["kind"] for event in actual] == [kind for _ in range(4) for kind in ["start","request","request","stop"]]
        assert all(row["provider_lifecycle_receipt"]["warmup"]["completed"] for row in warmed["runs"])
        assert warmed["warmups_per_endpoint"] == 2
        for mode in ["exit","wrong","hang","warmup_error"]:
            bad = spec("warmed_process" if mode=="warmup_error" else "cold_process",mode)
            before = set((root/"receipts").glob("owned-compare-*"))
            began = time.monotonic()
            try:
                run(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"owned failure {mode} was accepted")
            assert time.monotonic()-began < 5
            directory, = set((root/"receipts").glob("owned-compare-*"))-before
            receipts = list(directory.glob("*/receipt.json"))
            assert len(receipts) == 1
            receipt = json.loads(receipts[0].read_text())
            assert receipt["shutdown"]["complete"], (mode,receipt)
            assert not any(listening(endpoint(url)) for url in urls)
            if mode == "warmup_error":
                assert receipt["warmup"]["completed"] is False
                assert "measurement" not in receipt
        # Existing listeners are observed and preserved; no child/group signals.
        with socket.socket() as existing:
            existing.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            existing.bind(("127.0.0.1",ports[0]));existing.listen()
            try:
                run(spec())
            except ValueError:
                pass
            else:
                raise AssertionError("pre-existing listener was adopted")
            assert listening(("127.0.0.1",ports[0]))
        # A TERM-ignoring owned foreground process needs an actual bounded KILL.
        plan = LifecyclePlan(spec(mode="ignore_term"),dict(zip(["zerogravity","vllm"],urls)),root/"force")
        def model_ready():
            import urllib.request
            with urllib.request.urlopen(urls[0]+"/v1/models",timeout=1) as response:
                assert json.load(response)["data"][0]["id"] == "fixture"
        try:
            with plan.sample("zerogravity",model_ready):
                raise RuntimeError("synthetic measurement exception")
        except RuntimeError:
            pass
        receipt = json.loads(Path(plan.receipts[0]).read_text())
        assert receipt["shutdown"]["complete"] and "SIGKILL" in receipt["shutdown"]["signals"]
        assert receipt["failure_type"] == "RuntimeError"
        assert not any(listening(endpoint(url)) for url in urls)
        # Synthetic CID manager exercises direct stop argv; it runs no container.
        fake = root/"podman"
        fake.write_text("#!"+sys.executable+"\n"+r'''
import json,os,pathlib,signal,sys,uuid
state=pathlib.Path(sys.argv[2]);state.mkdir(exist_ok=True)
action,args=sys.argv[3],sys.argv[4:]
if action=="run":
 cid=uuid.uuid4().hex+uuid.uuid4().hex
 cidfile=pathlib.Path(args[args.index("--cidfile")+1])
 if not (state/"skip-cid").exists():
  descriptor=os.open(cidfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  with os.fdopen(descriptor,"w") as target:target.write(cid)
 (state/(cid+".json")).write_text(json.dumps(dict(pid=os.getpid())))
 command=args[args.index("--fixture-argv")+1:]
 os.execv(command[0],command)
elif action=="stop":
 cid=args[-1]
 (state/"stopped-cid").write_text(cid)
 pid=json.loads((state/(cid+".json")).read_text())["pid"]
 os.kill(pid,signal.SIGTERM)
 sys.exit(9 if (state/"fail-stop").exists() else 0)
else:sys.exit(8)
''')
        fake.chmod(0o700)
        state = root/"synthetic-cid-state"
        settings = spec()
        provider = settings["providers"]["zerogravity"]
        provider["argv"] = [str(fake),"--state",str(state),"run","--cidfile","{cidfile}","--fixture-argv"]+provider["argv"]
        provider["stop_argv"] = [str(fake),"--state",str(state),"stop","--ignore","--time","1","{owned_cid}"]
        plan = LifecyclePlan(settings,dict(zip(["zerogravity","vllm"],urls)),root/"cid")
        with plan.sample("zerogravity",model_ready) as owned:
            bound_cid = owned.receipt["owned_cid"]
            # A foreign later replacement cannot change the bound stop target.
            owned.cidfile.write_text("f"*64)
        assert (state/"stopped-cid").read_text() == bound_cid != "f"*64
        assert owned.receipt["shutdown"]["complete"]
        assert owned.receipt["shutdown"]["stop"]["owned_cid"] == bound_cid
        (state/"fail-stop").touch()
        plan = LifecyclePlan(settings,dict(zip(["zerogravity","vllm"],urls)),root/"cid-failure")
        try:
            with plan.sample("zerogravity",model_ready):
                pass
        except ValueError:
            pass
        else:
            raise AssertionError("failed owned CID stop was accepted")
        assert plan.blocked and plan.current is None
        receipt = json.loads(Path(plan.receipts[0]).read_text())
        assert receipt["shutdown"]["stop"]["returncode"] == 9
        assert receipt["shutdown"]["owned_process_group_gone"] and receipt["shutdown"]["endpoints_closed"]
        assert not receipt["shutdown"]["complete"]
        try:
            with plan.sample("vllm",lambda:None):
                raise AssertionError("second provider started despite failed cleanup")
        except ValueError:
            pass
        (state/"fail-stop").unlink()
        (state/"skip-cid").touch()
        plan = LifecyclePlan(settings,dict(zip(["zerogravity","vllm"],urls)),root/"cid-missing")
        try:
            with plan.sample("zerogravity",model_ready):
                raise AssertionError("container without fresh CID became ready")
        except ValueError:
            pass
        receipt = json.loads(Path(plan.receipts[0]).read_text())
        assert plan.blocked and receipt["shutdown"]["stop"]["status"] == "no_fresh_CID"
        assert receipt["shutdown"]["owned_process_group_gone"] and receipt["shutdown"]["endpoints_closed"]
        assert not receipt["shutdown"]["complete"]
        # Permanent receipt/storage failures cannot prevent physical cleanup.
        original_write = globals()["write_json"]
        (state/"skip-cid").unlink()
        for phase in ["spawn", "warmup", "measurement", "shutdown", "startup-cid"]:
            case = copy.deepcopy(settings) if phase in {"shutdown", "startup-cid"} else spec()
            if phase == "startup-cid":
                command = case["providers"]["zerogravity"]["argv"]
                command[command.index("normal")] = "wrong"
            plan = LifecyclePlan(case,dict(zip(["zerogravity","vllm"],urls)),root/("io-"+phase))
            injected = {}
            def failed_write(path,value):
                target = (phase == "spawn" and path.name == "receipt.json" and value.get("pid") is not None
                          or phase == "warmup" and path.name == "receipt.json" and value.get("warmup") is not None
                          or phase == "measurement" and path.name == "measurement.json"
                          or phase == "shutdown" and path.name == "receipt.json" and value.get("shutdown") is not None
                          or phase == "startup-cid" and path.name == "receipt.json"
                              and value.get("startup",{}).get("last_failures") and plan.current.cidfile.exists())
                if injected or target:
                    injected["owned"] = plan.current
                    raise OSError("synthetic persistent receipt write failure")
                original_write(path,value)
            globals()["write_json"] = failed_write
            try:
                try:
                    with plan.sample("zerogravity",model_ready) as owned:
                        if phase == "warmup":
                            from dataclasses import asdict
                            warm = comparison.__globals__["stream_one"](urls[0],"fixture","Hi.",1,3,None,request_id="storage-fixture-warmup")
                            assert warm.completed
                            owned.record_warmup(asdict(warm))
                        elif phase == "measurement":
                            measured = comparison.__globals__["bench"](urls[0],"fixture","Hi.",1,1,1,3,None)
                            assert measured["completed_requests"] == 1
                            owned.record_measurement(measured)
                except OSError as error:
                    assert "synthetic persistent" in str(error)
                else:
                    raise AssertionError(f"lost {phase} receipt evidence was accepted")
            finally:
                globals()["write_json"] = original_write
            owned = injected["owned"]
            shutdown = owned.receipt["shutdown"]
            assert shutdown["cleanup_complete"] and shutdown["owned_process_group_gone"] and shutdown["endpoints_closed"], (phase,shutdown)
            assert shutdown["readiness_reader_complete"] and not group_alive(owned.process.pid)
            assert owned.receipt["log"]["reader_complete"] and not owned.log.thread.is_alive()
            assert owned.process.stdout.closed and plan.blocked and plan.current is None
            assert shutdown["evidence_write_error"] == "OSError"
            if phase in {"shutdown", "startup-cid"}:
                assert shutdown["stop"]["returncode"] == 0
                assert (state/"stopped-cid").read_text() == owned.receipt["owned_cid"]
            count = len(plan.receipts)
            try:
                with plan.sample("vllm",lambda:None):
                    raise AssertionError("next provider started without receipt evidence")
            except ValueError:
                pass
            assert len(plan.receipts) == count
        # Strict parser/target checks execute no commands.
        malformed = []
        bad=spec();bad["unknown"]=1;malformed.append(bad)
        bad=spec();bad["startup_timeout_seconds"]=float("inf");malformed.append(bad)
        bad=spec();bad["providers"]["zerogravity"]["argv"]=["sh","-c","echo unsafe"];malformed.append(bad)
        bad=spec();bad["providers"]["zerogravity"]["stop_argv"]=["kill","-9","1"];malformed.append(bad)
        bad=spec();bad["providers"]["zerogravity"]["declarations"]["source_sha256"]="unknown";malformed.append(bad)
        for bad in malformed:
            try:validate(bad)
            except ValueError:pass
            else:raise AssertionError("malformed lifecycle spec accepted")
        valid=spec();valid["providers"]["zerogravity"]["argv"]=["podman","--root","/owned/storage","run","--cidfile","{cidfile}","--name","{owned_name}","image"]
        valid["providers"]["zerogravity"]["stop_argv"]=["podman","--root","/owned/storage","stop","--ignore","--time","1","{owned_cid}"]
        validate(valid)
        bad=copy.deepcopy(valid);bad["providers"]["zerogravity"]["stop_argv"][-1]="unrelated-container"
        try:validate(bad)
        except ValueError:pass
        else:raise AssertionError("foreign container stop target accepted")
        print("SERVING_OWNED_PROVIDER_LIFECYCLE_SOURCE_FREE_SELF_TEST_PASS")
