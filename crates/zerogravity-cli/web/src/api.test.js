import { test } from "node:test";
import assert from "node:assert/strict";
import { takeKey, eventDecoder } from "./api.js";

test("private fragment consumed without query or persistent local authority", () => {
  const data = new Map();
  const storage = { getItem: key => data.get(key), setItem: (key,value) => data.set(key,value), removeItem: key => data.delete(key) };
  const token = "a".repeat(64);
  let replaced;
  const history = { replaceState: (_, unused, path) => { replaced = path; } };
  assert.equal(takeKey({ hash: "#control=" + token, pathname: "/ui/" }, history, storage), token);
  assert.equal(replaced, "/ui/");
  assert.equal(takeKey({ hash: "", pathname: "/ui/" }, history, storage), token);
  assert.equal(takeKey({ hash: "#control=wrong", pathname: "/ui/" }, history, storage), "");
  assert.equal(data.size, 0);
});

test("coalesced event frames survive fragmented delivery and resync", () => {
  let refreshes = 0;
  const decode = eventDecoder(() => refreshes++);
  decode("event: re");
  decode("fresh\ndata: {}\n\n: keepalive\n\n");
  decode("event: resync\ndata: {}");
  decode("\n\n");
  assert.equal(refreshes, 2);
  assert.throws(() => decode("x".repeat(16385)), /bound/);
});

// Exact decimal identity comparison is independent of JavaScript number range.
import { newerId } from "./work-state.js";
test("action identity ordering never coerces u64 to Number", () => {
  assert.equal(newerId("9007199254740992","9007199254740993"),"9007199254740993");
  assert.equal(newerId("999","1000"),"1000");
  assert.equal(newerId(null,"18446744073709551615"),"18446744073709551615");
});
