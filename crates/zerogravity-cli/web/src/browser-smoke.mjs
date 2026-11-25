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
const children = [];
let ws;
function child(command, args, env) {
  const process = spawn(command, args, { cwd: root, env, stdio: ["ignore", "pipe", "pipe"] });
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
async function waitFor(condition, name) {
  const end = Date.now() + 12000;
  do {
    if (await condition()) return;
    await delay(100);
  } while (Date.now() < end);
  throw new Error("Browser condition did not settle: " + name);
}
try {
  const env = { ...process.env, HOME: root, XDG_CONFIG_HOME: join(root,"config"), XDG_CACHE_HOME: join(root,"cache") };
  const host = child(binary, ["--offline", "web"], env);
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
  // Optional local screenshot for visual review, never a credential or
  // arbitrary browser-history capture. No artifact is required for the test.
  if (process.env.ZG_WEB_SCREENSHOT) {
    const image = await bidi("browsingContext.captureScreenshot",{context});
    await writeFile(process.env.ZG_WEB_SCREENSHOT,Buffer.from(image.data,"base64"));
  }
  await bidi("session.end");
  ws.close();
  assert.equal((await fetch(origin + "/ui/")).status, 200);
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
