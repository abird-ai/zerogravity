#!/usr/bin/env python3
"""Source-free lifecycle evidence fixtures; no container/model/device runtime."""
import copy
import fcntl
import json
import os
from pathlib import Path
import runpy
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


LIFECYCLE = runpy.run_path(str(Path(__file__).with_name("serve-lifecycle.py")))
GLOBALS = LIFECYCLE["OwnedProvider"].close.__globals__
FIXTURE = r'''
import json,os,pathlib,signal,sys,time,uuid
state=pathlib.Path(sys.argv[2]);state.mkdir(exist_ok=True)
action,args=sys.argv[3],sys.argv[4:]
mode=(state/"mode").read_text()
if action=="run":
 cid=uuid.uuid4().hex+uuid.uuid4().hex
 cidfile=pathlib.Path(args[args.index("--cidfile")+1])
 signal.signal(signal.SIGTERM,lambda *_:sys.exit(137 if mode=="provider-exit137" else 0))
 (state/(cid+".json")).write_text(json.dumps(dict(pid=os.getpid())))
 if mode!="missing-cid":cidfile.write_text(cid)
 while True:time.sleep(.05)
elif action=="container":
 assert args[:3]==["inspect","--format","{{json .State}}"]
 cid=args[-1]
 assert len(cid)==64 and (state/(cid+".json")).exists()
 if mode=="inspect-absent":
  print("no such owned container",file=sys.stderr);sys.exit(125)
 if mode=="inspect-hang":time.sleep(30)
 stopped=(state/(cid+".stopped")).exists() and mode!="still-running"
 result=dict(Running=not stopped,ExitCode=137 if mode=="terminal-oom" else 0,
  OOMKilled=mode=="terminal-oom",Error="")
 if mode=="malformed-exit-status":result["ExitCode"]=False
 if mode=="malformed-running":result["Running"]="false"
 if mode=="malformed-oom":result["OOMKilled"]="false"
 print(json.dumps(result),flush=True)
elif action=="stop":
 cid=args[-1]
 assert args[:3]==["--ignore","--time","1"]
 data=json.loads((state/(cid+".json")).read_text())
 (state/"stopped-cid").write_text(cid)
 print("stop stdout",flush=True);print("stop stderr",file=sys.stderr,flush=True)
 if mode=="noisy":
  os.write(1,b"o"*131072);os.write(2,b"e"*131072)
 if mode in {"hang-kill","hang-term-zero","descendants"}:
  if mode=="hang-kill":signal.signal(signal.SIGTERM,signal.SIG_IGN)
  elif mode=="hang-term-zero":
   signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
   (state/(cid+".stop-ready")).write_text(str(os.getpid()))
  else:
   child=os.fork()
   if child==0:
    grandchild=os.fork()
    if grandchild==0:
     signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
     print("grandchild",flush=True)
     while True:time.sleep(.05)
    def child_term(*_):
     os.waitpid(grandchild,0);sys.exit(0)
    signal.signal(signal.SIGTERM,child_term)
    print("child",file=sys.stderr,flush=True)
    while True:time.sleep(.05)
   def parent_term(*_):
    os.waitpid(child,0);sys.exit(0)
   signal.signal(signal.SIGTERM,parent_term)
  while True:time.sleep(.05)
 os.kill(data["pid"],signal.SIGTERM)
 (state/(cid+".stopped")).write_text("stopped")
 sys.exit(9 if mode=="failure" else 137 if mode=="stop-exit137" else 0)
else:sys.exit(8)
'''


class LifecycleEvidence(unittest.TestCase):
    def test_log_reader_retains_eof_with_descriptor_above_select_limit(self):
        if resource.getrlimit(resource.RLIMIT_NOFILE)[0] <= 1100:
            self.skipTest("requires an already available descriptor above 1024")
        with tempfile.TemporaryDirectory() as temporary:
            read_fd, write_fd = os.pipe()
            try:
                high_fd = fcntl.fcntl(read_fd, fcntl.F_DUPFD_CLOEXEC, 1100)
            finally:
                os.close(read_fd)
            pipe = os.fdopen(high_fd, "rb", buffering=0)
            try:
                log = LIFECYCLE["BoundedLog"](pipe, Path(temporary)/"high-fd.log", 1024)
                os.write(write_fd, b"complete output")
                os.close(write_fd)
                write_fd = None
                result = log.finish(2)
                self.assertTrue(result["capture_complete"], result)
                self.assertTrue(result["eof"])
                self.assertIsNone(result["reader_error"])
                self.assertEqual(Path(result["path"]).read_bytes(), b"complete output")
                self.assertTrue(pipe.closed)
                self.assertFalse(log.thread.is_alive())
            finally:
                if write_fd is not None:
                    os.close(write_fd)
                pipe.close()

    def test_reference_command_evidence_requires_no_http_provider(self):
        for code in (0, 7):
            with self.subTest(returncode=code), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                observations = []
                def checkpoint(stage, record):
                    observations.append((stage, record["pid"], record["leader_reaped"]))
                    LIFECYCLE["write_json"](directory/"command.json", record)
                    return True
                record = LIFECYCLE["owned_command"](
                    [sys.executable, "-B", "-c",
                     "import os,sys;os.write(1,b'out');os.write(2,b'err');sys.exit("+str(code)+")"],
                    "reference-stop", "a"*64, 2, checkpoint,
                    directory=directory, container_name="owned-reference", log_limit_bytes=1024)
                self.assertEqual(observations[0], ("reference-stop_intent", None, False))
                self.assertEqual([row[0] for row in observations],
                                 ["reference-stop_intent", "reference-stop_spawn", "reference-stop_result"])
                self.assertEqual(record["returncode"], code)
                self.assertEqual(record["status"], "finished")
                self.assertTrue(record["evidence_complete"])
                self.assertTrue(record["result_persisted"])
                self.assertEqual(record["signals"], [])
                self.assertEqual(record["leader_reap_attempts"], 1)
                self.assertTrue(record["owned_process_group_gone"])
                self.assertEqual(Path(record["stdout"]["path"]).read_bytes(), b"out")
                self.assertEqual(Path(record["stderr"]["path"]).read_bytes(), b"err")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="zg-stop-evidence-")
        self.root = Path(self.temporary.name)
        self.fake = self.root/"podman"
        self.fake.write_text("#!"+sys.executable+"\n"+FIXTURE)
        self.fake.chmod(0o700)
        self.state = self.root/"state"
        self.state.mkdir()
        self.urls = {}
        for label in ["zerogravity","vllm"]:
            with socket.socket() as reserved:
                reserved.bind(("127.0.0.1",0))
                self.urls[label] = "http://127.0.0.1:"+str(reserved.getsockname()[1])
        while len(set(self.urls.values())) != 2:
            with socket.socket() as reserved:
                reserved.bind(("127.0.0.1",0))
                self.urls["vllm"] = "http://127.0.0.1:"+str(reserved.getsockname()[1])
        declaration = dict(runtime="synthetic-direct-child",source_sha256="a"*64,tokenizer_sha256="b"*64,
            numerical_profile="source-free",quantization="none",generation_policy="no inference",
            hardware="CPU subprocess fixture",resource_limits={"gpu":False})
        provider = dict(argv=[str(self.fake),"--state",str(self.state),"run","--cidfile","{cidfile}","--name","{owned_name}","fixture"],
            stop_argv=[str(self.fake),"--state",str(self.state),"stop","--ignore","--time","1","{owned_cid}"],
            declarations=declaration)
        self.spec = dict(schema=LIFECYCLE["SCHEMA"],policy="cold_process",startup_timeout_seconds=1,
            shutdown_timeout_seconds=1,log_limit_bytes=1024,providers={label:copy.deepcopy(provider) for label in self.urls})

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self, mode):
        (self.state/"mode").write_text(mode)
        return LIFECYCLE["LifecyclePlan"](self.spec,self.urls,self.root/("receipts-"+mode))

    @staticmethod
    def ready(plan):
        if not plan.current.cidfile.exists():
            raise ValueError("fixture has not written its CID")

    def sample(self, plan, body=lambda owned:None):
        with plan.sample("zerogravity",lambda:self.ready(plan)) as owned:
            body(owned)
        return owned

    def receipt(self, plan):
        return json.loads(Path(plan.receipts[0]).read_text())

    def assert_cleanup(self, shutdown):
        self.assertTrue(shutdown["cleanup_complete"],shutdown)
        self.assertTrue(shutdown["owned_process_group_gone"])
        self.assertTrue(shutdown["endpoints_closed"])
        self.assertTrue(shutdown["foreground"]["leader_reaped"])
        self.assertEqual(shutdown["foreground"]["leader_reap_attempts"],1)

    def test_success_intent_streams_clocks_cid_and_owned_signals(self):
        self.spec["providers"]["zerogravity"]["redact_arg_indices"] = [2]
        plan = self.plan("normal")
        original = subprocess.Popen
        launches = []
        def checked(command,**kwargs):
            if command[3] == "stop":
                intent = self.receipt(plan)["shutdown"]["stop"]
                self.assertEqual(intent["status"],"intent")
                self.assertIsNone(intent["pid"])
                self.assertTrue(intent["intent_persisted"])
                launches.append(intent)
            return original(command,**kwargs)
        with patch.object(subprocess,"Popen",checked):
            owned = self.sample(plan,lambda item:item.cidfile.write_text("f"*64))
        shutdown = owned.receipt["shutdown"]
        self.assert_cleanup(shutdown)
        self.assertTrue(shutdown["accepted"])
        self.assertTrue(shutdown["graceful_terminal_outcome"])
        self.assertEqual(len(launches),1)
        stop = shutdown["stop"]
        self.assertEqual(stop["owned_cid"],(self.state/"stopped-cid").read_text())
        self.assertNotEqual(stop["owned_cid"],"f"*64)
        self.assertEqual(stop["owned_container_name"],owned.name)
        self.assertEqual(stop["pid"],stop["pgid"])
        self.assertEqual(stop["returncode"],0)
        self.assertEqual(stop["argv"][2],"[REDACTED]")
        self.assertTrue(all(item["argv"][2] == "[REDACTED]" for item in shutdown["container_state"]))
        self.assertFalse(stop["timed_out"])
        self.assertEqual(stop["signals"],[])
        self.assertTrue(stop["leader_reaped"])
        self.assertEqual(stop["leader_reap_attempts"],1)
        self.assertLessEqual(stop["intent"]["monotonic_ns"],stop["start"]["monotonic_ns"])
        self.assertLessEqual(stop["start"]["monotonic_ns"],stop["end"]["monotonic_ns"])
        for stream in ["stdout","stderr"]:
            capture = stop[stream]
            self.assertEqual(Path(capture["path"]).read_bytes(),("stop "+stream+"\n").encode())
            self.assertEqual(Path(capture["path"]).stat().st_mode & 0o777,0o600)
            self.assertTrue(capture["capture_complete"])
        self.assertEqual(shutdown["container_oom"],"observed_false")
        self.assertTrue(shutdown["container_terminal_state_observed"])

    def test_failures_timeouts_and_exit137_do_not_claim_graceful(self):
        for mode in ["failure","stop-exit137","provider-exit137","hang-term-zero","hang-kill","descendants"]:
            with self.subTest(mode=mode):
                plan = self.plan(mode)
                with self.assertRaises(ValueError):
                    self.sample(plan)
                shutdown = self.receipt(plan)["shutdown"]
                self.assert_cleanup(shutdown)
                self.assertFalse(shutdown["accepted"])
                self.assertFalse(shutdown["graceful_terminal_outcome"])
                self.assertTrue(plan.blocked)
                stop = shutdown["stop"]
                self.assertTrue(stop["owned_process_group_gone"])
                self.assertTrue(stop["leader_reaped"])
                self.assertEqual(stop["leader_reap_attempts"],1)
                if mode.startswith("hang") or mode == "descendants":
                    self.assertTrue(stop["timed_out"])
                    self.assertEqual(stop["status"],"timed_out")
                    self.assertTrue(stop["all_nonzero_signals_before_reap"])
                    self.assertTrue(all(item["leader_unreaped"] for item in stop["signals"]))
                    self.assertEqual(stop["returncode"],-signal.SIGKILL if mode=="hang-kill" else 0)
                if mode == "stop-exit137":
                    self.assertEqual(stop["terminal_outcome"]["kind"],"exit")
                    self.assertIsNone(stop["terminal_outcome"]["signal"])
                    self.assertEqual(stop["terminal_outcome"]["oom"],"not_established")
                if mode == "provider-exit137":
                    self.assertTrue(shutdown["complete"])
                    self.assertEqual(shutdown["provider_terminal_outcome"]["kind"],"exit")
                    self.assertEqual(shutdown["provider_terminal_outcome"]["oom"],"not_established")

    def test_bounded_output_is_explicitly_incomplete_and_rejected(self):
        plan = self.plan("noisy")
        with self.assertRaises(ValueError):
            self.sample(plan)
        shutdown = self.receipt(plan)["shutdown"]
        self.assert_cleanup(shutdown)
        self.assertFalse(shutdown["complete"])
        for stream in ["stdout","stderr"]:
            capture = shutdown["stop"][stream]
            self.assertGreater(capture["bytes_seen"],1024)
            self.assertEqual(capture["bytes_retained"],1024)
            self.assertTrue(capture["truncated"])
            self.assertTrue(capture["eof"])
            self.assertFalse(capture["capture_complete"])

    def test_private_log_write_failure_does_not_skip_cleanup(self):
        plan = self.plan("normal")
        original = GLOBALS["BoundedLog"]
        class LostLog(original):
            def __init__(self,pipe,path,limit):
                if path.name == "stop.stderr.log":
                    path.mkdir()
                super().__init__(pipe,path,limit)
        with patch.dict(GLOBALS,BoundedLog=LostLog):
            with self.assertRaises(ValueError):
                self.sample(plan)
        shutdown = self.receipt(plan)["shutdown"]
        self.assert_cleanup(shutdown)
        capture = shutdown["stop"]["stderr"]
        self.assertEqual(capture["reader_error"],"FileExistsError")
        self.assertTrue(capture["eof"])
        self.assertFalse(capture["capture_complete"])
        self.assertFalse(shutdown["accepted"])

    def test_missing_eof_rejects_full_capture_without_leaking_reader(self):
        read_fd,write_fd = os.pipe()
        pipe = os.fdopen(read_fd,"rb",buffering=0)
        log = LIFECYCLE["BoundedLog"](pipe,self.root/"missing-eof.log",1024)
        try:
            os.write(write_fd,b"output without EOF")
            result = log.finish(.1)
            self.assertFalse(result["eof"])
            self.assertFalse(result["capture_complete"])
            self.assertTrue(result["reader_complete"])
            self.assertFalse(log.thread.is_alive())
            self.assertTrue(pipe.closed)
            self.assertEqual(result["bytes_retained"],len(b"output without EOF"))
        finally:
            os.close(write_fd)

    def test_stop_launch_failure_is_unknown_and_still_cleans_provider(self):
        plan = self.plan("normal")
        original = subprocess.Popen
        def missing(command,**kwargs):
            if command[3] == "stop":
                raise FileNotFoundError("synthetic missing stop executable")
            return original(command,**kwargs)
        with patch.object(subprocess,"Popen",missing):
            with self.assertRaises(ValueError):
                self.sample(plan)
        shutdown = self.receipt(plan)["shutdown"]
        self.assert_cleanup(shutdown)
        stop = shutdown["stop"]
        self.assertEqual(stop["status"],"program_not_found")
        self.assertIsNone(stop["pid"])
        self.assertIsNone(stop["returncode"])
        self.assertFalse(stop["evidence_complete"])
        self.assertFalse(shutdown["accepted"])

    def test_stop_receipt_failure_never_skips_owned_cleanup(self):
        for phase in ["intent","running","finished"]:
            with self.subTest(phase=phase):
                plan = self.plan("normal")
                original = GLOBALS["write_json"]
                injected = []
                def failed(path,value):
                    stop = (value.get("shutdown") or {}).get("stop") or {}
                    if injected or path.name == "receipt.json" and stop.get("status") == phase:
                        injected.append(plan.current)
                        raise OSError("synthetic stop receipt failure")
                    return original(path,value)
                with patch.dict(GLOBALS,write_json=failed):
                    with self.assertRaises(OSError):
                        self.sample(plan)
                owned = injected[0]
                shutdown = owned.receipt["shutdown"]
                self.assert_cleanup(shutdown)
                self.assertEqual(shutdown["stop"]["returncode"],0)
                self.assertEqual(shutdown["evidence_write_error"],"OSError")
                self.assertFalse(shutdown["accepted"])
                self.assertTrue(plan.blocked)
                self.assertEqual(shutdown["stop"]["intent_persisted"],phase != "intent")

    def test_post_signal_exception_preserves_escalation_and_rejects_recovery(self):
        for error_type in [KeyboardInterrupt,RuntimeError]:
            with self.subTest(error_type=error_type.__name__):
                plan = self.plan("hang-term-zero")
                original = os.killpg
                injected = []
                def interrupted(pid,sig):
                    stop = (plan.current.receipt.get("shutdown") or {}).get("stop") or {}
                    if not injected and sig == signal.SIGTERM and pid == stop.get("pid"):
                        marker = self.state/(stop["owned_cid"]+".stop-ready")
                        deadline = time.monotonic()+5
                        while (not marker.exists() or marker.read_text() != str(pid)) and time.monotonic() < deadline:
                            time.sleep(.01)
                        self.assertTrue(marker.exists() and marker.read_text() == str(pid))
                        original(pid,sig)
                        injected.append(pid)
                        # Raise only after the intended clean exit is observed,
                        # still unreaped. Recovery must retain the earlier TERM
                        # evidence, not race Python finalization with another.
                        observed = None
                        while time.monotonic() < deadline:
                            observed = os.waitid(os.P_PID,pid,os.WEXITED | os.WNOHANG | os.WNOWAIT)
                            if observed is not None:
                                break
                            time.sleep(.01)
                        self.assertIsNotNone(observed)
                        self.assertEqual((observed.si_code,observed.si_status),(os.CLD_EXITED,0))
                        raise error_type("synthetic interruption after owned TERM dispatch")
                    return original(pid,sig)
                with patch.object(os,"killpg",interrupted):
                    with self.assertRaises(ValueError):
                        self.sample(plan)
                shutdown = self.receipt(plan)["shutdown"]
                stop = shutdown["stop"]
                self.assertEqual(len(injected),1)
                self.assert_cleanup(shutdown)
                self.assertTrue(shutdown["foreground"]["complete"])
                self.assertTrue(stop["owned_process_group_gone"])
                self.assertTrue(stop["leader_reaped"])
                self.assertEqual(stop["leader_reap_attempts"],1)
                self.assertEqual(stop["returncode"],0)
                self.assertEqual(stop["status"],"command_error")
                self.assertEqual(stop["error_type"],error_type.__name__)
                self.assertTrue(stop["timed_out"])
                self.assertEqual(stop["terminal_before_cleanup"]["kind"],"unknown")
                self.assertEqual(stop["signals"][0]["signal"],"SIGTERM")
                self.assertIsNone(stop["signals"][0]["sent"])
                self.assertTrue(stop["errors"])
                if error_type is KeyboardInterrupt:
                    self.assertEqual(len(stop["retirement_attempts"]),2)
                    self.assertTrue(stop["recovery_observations"])
                self.assertFalse(shutdown["graceful_terminal_outcome"])
                self.assertFalse(shutdown["accepted"])
                self.assertTrue(plan.blocked)

    def test_container_state_unavailable_is_not_oom_false(self):
        for mode in ["inspect-absent","inspect-hang"]:
            with self.subTest(mode=mode):
                plan = self.plan(mode)
                owned = self.sample(plan)
                shutdown = owned.receipt["shutdown"]
                self.assert_cleanup(shutdown)
                self.assertEqual(shutdown["container_oom"],"not_established")
                self.assertFalse(shutdown["container_terminal_state_observed"])
                self.assertTrue(all(item["observation"] == "unavailable" for item in shutdown["container_state"]))
                if mode == "inspect-hang":
                    self.assertTrue(all(item["timed_out"] for item in shutdown["container_state"]))
        plan = self.plan("terminal-oom")
        with self.assertRaises(ValueError):
            self.sample(plan)
        shutdown = self.receipt(plan)["shutdown"]
        self.assert_cleanup(shutdown)
        self.assertEqual(shutdown["container_oom"],"observed_true")
        self.assertFalse(shutdown["graceful_terminal_outcome"])

    def test_live_post_stop_state_and_observability_failure(self):
        plan = self.plan("still-running")
        with self.assertRaises(ValueError):
            self.sample(plan)
        shutdown = self.receipt(plan)["shutdown"]
        self.assert_cleanup(shutdown)
        self.assertFalse(shutdown["graceful_terminal_outcome"])
        self.assertTrue(shutdown["container_state"][-1]["state"]["Running"])
        plan = self.plan("normal")
        def unavailable(*_):
            raise RecursionError("synthetic State observation failure")
        with patch.object(LIFECYCLE["OwnedProvider"],"inspect_container",unavailable):
            owned = self.sample(plan)
        shutdown = owned.receipt["shutdown"]
        self.assert_cleanup(shutdown)
        self.assertEqual(shutdown["stop"]["returncode"],0)
        self.assertEqual(shutdown["container_oom"],"not_established")
        self.assertTrue(all(item["observation"] == "unavailable" for item in shutdown["container_state"]))

    def test_malformed_state_never_establishes_graceful_or_oom_absence(self):
        for mode in ["malformed-exit-status","malformed-running","malformed-oom"]:
            with self.subTest(mode=mode):
                plan = self.plan(mode)
                with self.assertRaises(ValueError):
                    self.sample(plan)
                shutdown = self.receipt(plan)["shutdown"]
                self.assert_cleanup(shutdown)
                self.assertEqual(shutdown["container_oom"],"not_established")
                self.assertFalse(shutdown["container_terminal_state_observed"])
                self.assertFalse(shutdown["graceful_terminal_outcome"])
                for item in shutdown["container_state"]:
                    self.assertIsNone(item["state"])
                    self.assertEqual(item["observation"],"unavailable")
                    self.assertEqual(item["parse_error_type"],"ValueError")

    def test_reaped_or_unknown_child_never_authenticates_group_signal(self):
        child = subprocess.Popen([sys.executable,"-B","-c","pass"],start_new_session=True)
        self.assertEqual(child.wait(timeout=2),0)
        original = os.killpg
        attempted = []
        def checked(pid,sig):
            if sig:
                attempted.append((pid,sig))
            return original(pid,sig)
        record = {}
        with patch.object(os,"killpg",checked):
            self.assertFalse(LIFECYCLE["retire_child"](child,record))
        self.assertEqual(attempted,[])
        self.assertEqual(record["leader_reap_attempts"],0)
        self.assertIsNone(record["returncode"])
        self.assertEqual(record["terminal_outcome"]["kind"],"unknown")
        self.assertEqual(record["terminal_outcome"]["oom"],"not_established")


if __name__ == "__main__":
    unittest.main(verbosity=2)
