// Explicit developer smoke against the real embedded binary and installed
// Firefox. No browser or Node is required by ordinary Rust release builds.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve, join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";

const binary = resolve(process.argv[2]);
const root = await mkdtemp(join(tmpdir(), "zg-browser-"));
const residentRoot = process.argv[3] === "--resident" ? resolve(process.argv[4]) : null;
const expected = process.argv[5];
const downloadFixture = process.argv[3] === "--download-fixture";
const children = [];
let ws;
function child(command, args, env, input = "ignore") {
  const process = spawn(command, args, { cwd: root, env, stdio: [input, "pipe", "pipe"] });
  children.push(process);
  return process;
}
function line(process, pattern) {
  return new Promise((resolve, reject) => {
    let buffers = ["", ""];
    const timeout = setTimeout(() => done(new Error("Process startup timeout")), 20000);
    const exit = () => done(new Error("Process exited before reporting ready"));
    const error = () => done(new Error("Process could not start"));
    const readers = [process.stdout, process.stderr];
    const callbacks = readers.map((reader, index) => data => {
      buffers[index] = (buffers[index] + data.toString()).slice(-16384);
      const match = buffers[index].match(pattern);
      if (match) done(null, match[1]);
    });
    function done(failure, value) {
      clearTimeout(timeout);
      process.off("exit", exit); process.off("error", error);
      readers.forEach((reader,index) => reader.off("data", callbacks[index]));
      // Do not retain or print browser logs containing launch capabilities.
      readers.forEach(reader => reader.resume());
      failure ? reject(failure) : resolve(value);
    }
    readers.forEach((reader,index) => reader.on("data", callbacks[index]));
    process.once("exit", exit); process.once("error", error);
  });
}
async function waitFor(condition, name, timeout = 12000) {
  const end = Date.now() + timeout;
  do {
    if (await condition()) return;
    await delay(100);
  } while (Date.now() < end);
  throw new Error("Browser condition did not settle: " + name);
}
try {
  const env = { ...process.env, HOME: root, XDG_CONFIG_HOME: join(root,"config"), XDG_CACHE_HOME: join(root,"cache") };
  const hostEnv = residentRoot ? {
    ...env, HOME: residentRoot, XDG_CONFIG_HOME: join(residentRoot,"config"), XDG_CACHE_HOME: join(residentRoot,"cache"),
    ZEROGRAVITY_API_KEY: "fixture-inference-only",
  } : env;
  const host = downloadFixture
    ? child(binary, ["--ignored", "--exact", "pull::monitor::tests::monitor_child", "--nocapture"], {
        ...hostEnv, ZG_MONITOR_FIXTURE:join(root,"cache/abird/zerogravity/models"), ZG_MONITOR_MODE:"web",
      }, "pipe")
    : child(binary, residentRoot ? ["--offline","--profile","local","web"] : ["--offline", "web"], hostEnv);
  const privateUrl = await line(host, /Web UI: (http:\/\/127\.0\.0\.1:\d+\/ui\/#control=[a-f0-9]{64})/);
  const origin = new URL(privateUrl).origin;
  const profile = join(root,"firefox");
  await mkdir(profile);
  await writeFile(join(profile,"user.js"), [
    'user_pref("browser.shell.checkDefaultBrowser", false);',
    'user_pref("browser.startup.page", 0);',
    'user_pref("browser.newtabpage.enabled", false);',
    'user_pref("app.update.auto", false);',
    'user_pref("datareporting.healthreport.uploadEnabled", false);',
    'user_pref("toolkit.telemetry.enabled", false);',
  ].join("\n"));
  const browser = child(process.env.FIREFOX || "firefox", ["--headless", "--no-remote", "--profile", profile, "--remote-debugging-port", "0", "about:blank"], env);
  const endpoint = await line(browser, /WebDriver BiDi listening on (ws:\/\/127\.0\.0\.1:\d+)/);
  ws = new WebSocket(endpoint + "/session");
  await new Promise((resolve,reject) => {
    const timer = setTimeout(() => reject(new Error("BiDi connection timeout")), 10000);
    ws.addEventListener("open", () => { clearTimeout(timer); resolve(); }, {once:true});
    ws.addEventListener("error", () => { clearTimeout(timer); reject(new Error("BiDi connection failed")); }, {once:true});
  });
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", event => {
    const message = JSON.parse(event.data);
    const call = pending.get(message.id);
    if (call) {
      pending.delete(message.id);
      clearTimeout(call.timer);
      if (message.type === "error") call.reject(new Error("BiDi " + message.error));
      else call.resolve(message.result);
    }
  });
  function bidi(method, params={}) {
    return new Promise((resolve,reject) => {
      const request = ++id;
      const timer = setTimeout(() => { pending.delete(request); reject(new Error("BiDi timeout: " + method)); }, 15000);
      pending.set(request,{resolve,reject,timer});
      ws.send(JSON.stringify({id:request,method,params}));
    });
  }
  await bidi("session.new",{capabilities:{alwaysMatch:{}}});
  const tree = await bidi("browsingContext.getTree");
  const context = tree.contexts[0].context;
  async function evaluate(expression) {
    const result = await bidi("script.evaluate", {expression, target:{context}, awaitPromise:true});
    if (result.type !== "success") throw new Error("Browser script failed");
    return result.result.value;
  }
  const viewport = (width,height) => bidi("browsingContext.setViewport",{context,viewport:{width,height},devicePixelRatio:1});
  await viewport(1440,900);
  await bidi("browsingContext.navigate",{context,url:privateUrl,wait:"complete"});
  await waitFor(() => evaluate("document.body.innerText.includes('Local process') && document.body.innerText.includes('connected')"), "authenticated Home");
  assert.equal(await evaluate("location.hash"), "");
  assert.equal(await evaluate("document.querySelector('h1').textContent"), "Home");
  assert.equal(await evaluate("document.body.innerText.includes('offline')"), true);
  for (const [width,height] of [[1440,900],[900,700],[390,844]]) {
    await viewport(width,height);
    await waitFor(() => evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), "responsive width " + width);
  }
  await evaluate("document.querySelector('[aria-label=\"Toggle navigation\"]').click()");
  await evaluate("document.querySelector('nav button[title=\"Running\"]').click()");
  assert.equal(await evaluate("document.querySelector('h1').textContent"), "Running");
  assert.equal(await evaluate("document.body.innerText.includes('No model owners')"), true);
  await evaluate("document.dispatchEvent(new KeyboardEvent('keydown',{key:'F1',bubbles:true}))");
  await waitFor(() => evaluate("!!document.querySelector('[role=dialog]')"), "palette");
  await evaluate("(()=>{let input=document.querySelector('[aria-label=\"Search views\"]');input.value='serve';input.dispatchEvent(new Event('input',{bubbles:true}));input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));return true})()");
  assert.equal(await evaluate("document.querySelector('h1').textContent"), "Serve");
  assert.equal(await evaluate("!!document.querySelector('[role=dialog]')"), false);
  await viewport(1440,900);
  assert.equal(await evaluate("document.querySelector('h1').textContent"), "Serve");
  await bidi("browsingContext.reload",{context,wait:"complete"});
  await waitFor(() => evaluate("document.body.innerText.includes('Local process') && document.body.innerText.includes('connected')"), "tab-local reload");
  await evaluate("Array.from(document.querySelectorAll('footer button')).find(button=>button.textContent==='Lock').click()");
  await waitFor(() => evaluate("document.body.innerText.includes('Unlock local control')"), "lock");
  assert.equal(await evaluate("sessionStorage.getItem('zg-control-session') === null"), true);
  await bidi("browsingContext.navigate",{context,url:privateUrl,wait:"complete"});
  await waitFor(() => evaluate("document.body.innerText.includes('Local process')"), "reopen private URL");
  const click = async label => {
    const expression = "(()=>{const b=Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()===" + JSON.stringify(label) + ");if(!b || b.disabled)return false;b.click();return true})()";
    await waitFor(() => evaluate(expression), "enabled " + label, 90000);
  };
  const fill = (selector, value) => evaluate("(()=>{const field=document.querySelector(" + JSON.stringify(selector) + ");field.value=" + JSON.stringify(value) + ";field.dispatchEvent(new Event('input',{bubbles:true}));return true})()");
  const go = name => evaluate("document.querySelector(" + JSON.stringify('nav button[title="' + name + '"]') + ").click()");
  const contains = text => evaluate("document.body.innerText.includes(" + JSON.stringify(text) + ")");
  await fill("#quick-source", "/tmp/invalid-browser-model.gguf");
  await click("Load model");
  await waitFor(() => contains("The control action or input was refused"), "typed POST rejection");
  await go("Models");
  await waitFor(async () => await contains("Model inventory") && await contains("Cached") && !await contains("Waiting for cold model inventory"), "cold model shelf");
  await click("Catalog");
  await waitFor(() => contains("qwen3.5-0.8b"), "catalog inventory");
  await fill("#model-search", "qwen3.5-0.8b");
  await waitFor(() => evaluate("document.querySelectorAll('.entries .row').length === 1"), "catalog search");
  await viewport(390,844);
  await waitFor(() => evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), "catalog responsive width");
  await viewport(1440,900);
  await click("Partial");
  await waitFor(() => contains("No matching observations."), "partial shelf");
  await go("Devices");
  await waitFor(async () => await contains("cpu:0") && await contains("Physical inventory"), "physical owner inventory");
  await go("Downloads");
  if (downloadFixture) {
    await waitFor(() => evaluate("document.querySelector('.panel-title .state')?.textContent==='paused'"), "original safe pause");
    await click("Resume download");
    await waitFor(() => evaluate("document.querySelector('progress')?.value > 0"), "accepted transfer prefix");
    await click("Pause download");
    await waitFor(() => evaluate("document.querySelector('.panel-title .state')?.textContent==='paused'"), "range checkpoint");
    await click("Resume download");
    await waitFor(() => evaluate("document.querySelector('.panel-title .state')?.textContent==='durable'"), "durable original publication");
    await go("Models");
    await waitFor(() => contains("fixture/monitor"), "published cache observation");
    console.log("Browser download: original operation pause, exact-range resume, durable publication and cold cache observation");
  } else {
    await waitFor(async () => await contains("No download operation") || await contains("Acquisition is not compiled"), "empty transfer");
  }
  await go("Models");
  if (await evaluate("!!document.querySelector('#pull-source')")) {
    await fill("#pull-source", "hf:fixture/model@" + "1".repeat(40));
    await fill("#pull-selection", "model.gguf");
    await click("Preview pull");
    await waitFor(async () => await contains("1 exact files") && await contains("model.gguf"), "pinned acquisition preview");
    await waitFor(() => evaluate("Array.from(document.querySelectorAll('button')).some(b=>b.textContent.trim()==='Confirm pull' && b.disabled)"), "offline consent refusal");
    await fill("#pull-selection", "quant:Q4_K_M");
    await waitFor(() => evaluate(`!document.querySelector('[aria-label="Confirm acquisition"]')`), "changed draft invalidates consent");
    await click("Preview pull");
    await waitFor(() => contains("GGUF quant: Q4_K_M"), "exact quant preview");
  }
  if (residentRoot) {
    await click("Cached");
    await waitFor(() => contains("fixture/serve"), "original cached snapshot shelf");
    await click("Inspect");
    await waitFor(() => contains("Manifest identity checked"), "exact digest inspection");
    await click("Verify payloads");
    await waitFor(() => contains("Payloads verified"), "original digest verification", 90000);
    await fill("#cache-alias", "web-fixture");
    await click("Create alias");
    await waitFor(() => contains("by-name:web-fixture"), "non-replacing alias");
    await click("Preview removal");
    await waitFor(() => contains("Removal preview; no mutation"), "removal preview");
    await click("Remove snapshot");
    await click("Cancel");
    assert.equal(await contains("fixture/serve"), true);
    await click("Choose device");
    await waitFor(() => contains("cpu:0"), "cached source device selection");
    await waitFor(() => evaluate("document.querySelector('#device-source')?.value.length===64"), "exact source carried from Models");
    await evaluate("Array.from(document.querySelectorAll('.entries .row')).find(b=>b.textContent.includes('cpu:0')).click()");
    await click("Check source / device");
    await waitFor(() => contains("Cold source/device policy accepted"), "original resident device check", 90000);
    await click("Load on device");
    await waitFor(() => contains("Model route set loaded"), "exact device load", 90000);
    await go("Running");
    await click("Unload owner");
    await click("Confirm");
    await waitFor(() => contains("No model owners"), "device-selected owner retired", 90000);
    console.log("Browser library: cached/catalog/partial shelves, exact inspect/verify/alias/removal preview, offline consent, physical device check and load");
  }
  await go("Home");
  if (residentRoot) {
    // Real action buttons, original cached fixture, actual inference output.
    // Wait for the authoritative empty route phase after retirement.
    // No test transport or injected frontend state.
    await waitFor(() => evaluate("!!document.querySelector('#quick-source')"), "empty route phase");
    await fill("#quick-source", "");
    await click("Load model");
    await waitFor(() => contains("Model route set loaded"), "model load", 90000);
    await waitFor(() => evaluate("!!document.querySelector('#quick-owner')"), "ready owner", 90000);
    await fill("#quick-prompt", "Continue: 1, 2, 3,");
    assert.equal(await evaluate("document.querySelector('.check input').checked"), false);
    await click("Run");
    await waitFor(() => evaluate("!!document.querySelector('.response-text')"), "streamed response", 90000);
    await waitFor(() => evaluate("document.querySelector('.response-text')?.textContent===" + JSON.stringify(expected)), "exact original output", 90000);
    await waitFor(() => evaluate("!!document.querySelector('.action-message') && document.querySelector('.action-message').textContent.includes('output tokens')"), "original terminal usage", 90000);
    await go("Serve");
    await fill("#listen-address", "127.0.0.1:0");
    await click("Start inference API");
    await waitFor(() => evaluate("!!document.querySelector('.endpoint') && document.querySelector('.endpoint').textContent.endsWith('/v1')"), "listener started", 90000);
    const inference = await evaluate("document.querySelector('.endpoint').textContent.trim()");
    const rejected = await fetch(inference + "/models");
    assert.equal(rejected.status, 401);
    const controlKey = new URLSearchParams(new URL(privateUrl).hash.slice(1)).get("control");
    const wrongAuthority = await fetch(inference + "/models", {headers:{Authorization:"Bearer " + controlKey}});
    assert.equal(wrongAuthority.status, 401);
    const models = await fetch(inference + "/models", {headers:{Authorization:"Bearer fixture-inference-only"}});
    assert.equal(models.status, 200);
    assert.deepEqual((await models.json()).data.map(model=>model.id), ["configured"]);
    await click("Stop inference API");
    await click("Confirm");
    await waitFor(() => contains("Listener stopped; loaded models retained"), "listener stopped", 90000);
    await go("Running");
    await waitFor(() => evaluate("!!document.querySelector('#owner-prompt')"), "running owner");
    await click("Unload owner");
    await click("Cancel");
    assert.equal(await evaluate("!!document.querySelector('#owner-prompt')"), true);
    await click("Unload owner");
    await click("Confirm");
    await waitFor(() => contains("No model owners"), "owner retired", 90000);
    await go("Home");
    await waitFor(() => evaluate("!!document.querySelector('#quick-source')"), "empty runtime");
    await fill("#quick-prompt", "Continue: 1, 2, 3,");
    await click("Load and run");
    await waitFor(() => evaluate("document.querySelector('.response-text')?.textContent===" + JSON.stringify(expected)), "combined load and run", 90000);
    await waitFor(() => evaluate("!!document.querySelector('.action-message') && document.querySelector('.action-message').textContent.includes('output tokens')"), "combined terminal usage", 90000);
    await bidi("browsingContext.reload",{context,wait:"complete"});
    await waitFor(() => evaluate("document.querySelector('.response-text')?.textContent===" + JSON.stringify(expected)), "retained exact Web receipt after reload", 90000);
    console.log("Browser resident: actual cached load/invoke, original output and usage, authenticated inference start/stop, confirmed unload, combined load/run and retained receipt");
  }
  // Optional local screenshot for visual review, never a credential or
  // arbitrary browser-history capture. No artifact is required for the test.
  if (process.env.ZG_WEB_SCREENSHOT) {
    const image = await bidi("browsingContext.captureScreenshot",{context});
    await writeFile(process.env.ZG_WEB_SCREENSHOT,Buffer.from(image.data,"base64"));
  }
  await bidi("session.end");
  ws.close();
  assert.equal((await fetch(origin + "/ui/")).status, 200);
  if (downloadFixture) host.stdin.end("done\n");
  else host.kill("SIGTERM");
  await Promise.race([new Promise(resolve => host.once("exit",resolve)), delay(15000,undefined,{ref:false})]);
  assert.equal(host.exitCode, 0, "foreground host did not retire successfully");
  console.log("Embedded browser: authenticated Home, desktop/tablet/mobile, navigation/palette, reload/lock, browser retirement independent of host");
} finally {
  ws?.close();
  for (const process of children.reverse()) {
    if (process.exitCode === null && process.signalCode === null) {
      process.kill("SIGTERM");
      await Promise.race([new Promise(resolve => process.once("exit",resolve)), delay(5000)]);
      if (process.exitCode === null && process.signalCode === null) {
        process.kill("SIGKILL");
        await new Promise(resolve => process.once("exit",resolve));
      }
    }
  }
  await rm(root,{recursive:true,force:true});
}
