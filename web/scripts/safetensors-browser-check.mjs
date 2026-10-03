// Runs the safetensors inference probe in a real browser (Chromium via
// playwright): the page downloads the checkpoint safetensors.json pins from the
// Hugging Face Hub through hf-hub's JavaScript driver (js_of_ocaml build, the
// browser host in memory, SHA-256 from Web Crypto), checks it against the pin,
// and runs inference on it. Its stdout, which js_of_ocaml sends to the console,
// must equal the native golden line for line.
//
// The Hub is reached through hf-hub's own example proxy (javascript/serve.cjs):
// a browser cannot read the metadata headers of the Hub's redirecting HEAD. This
// needs the network (HF_ENDPOINT selects another upstream).
//
// usage: node scripts/safetensors-browser-check.mjs <safetensors_probe.bc.js> \
//          <model_dir> <input.pt> <golden.txt>
// Set PLAYWRIGHT_BROWSERS_PATH to the playwright browser cache.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { chromium } from "@playwright/test";

const [probe, modelDir, input, golden] = process.argv.slice(2);
if (!probe || !modelDir || !input || !golden) {
  console.error("usage: safetensors-browser-check.mjs <safetensors_probe.bc.js> <model_dir> <input.pt> <golden.txt>");
  process.exit(2);
}

const hfHub = path.resolve("../vendored/ocaml-hf-hub/javascript");
const { createServer } = createRequire(import.meta.url)(path.join(hfHub, "serve.cjs"));
const proxy = createServer({ endpoint: process.env.HF_ENDPOINT || "https://huggingface.co" });

const modelFiles = new Set([
  "models/model.json",
  "models/safetensors.json",
  "data/weights/model_weights_config.json",
  "data/constants/model_constants_config.json",
]);

const driver = `
const text = async (rel) => {
  const r = await fetch("/model/" + rel);
  return r.ok ? await r.text() : null;
};
window.runProbe = async () => {
  const options = {
    program: await text("models/model.json"),
    weights: await text("data/weights/model_weights_config.json"),
    constants: await text("data/constants/model_constants_config.json"),
    map: await text("models/safetensors.json"),
    input: new Uint8Array(await (await fetch("/input.pt")).arrayBuffer()),
    cacheDir: "/hub-cache",
    endpoint: location.origin + "/hub",
  };
  return new Promise((resolve) => safetensorsProbe(options, resolve));
};`;

const page = `<!doctype html><meta charset=utf-8><title>safetensors</title>
<script src="/fetch.js"></script><script src="/browser-host.js"></script>
<script src="/probe.js"></script><script src="/driver.js"></script>`;

const script = (res, body) => { res.writeHead(200, { "Content-Type": "text/javascript" }); res.end(body); };

const server = http.createServer((req, res) => {
  const url = req.url.split("?")[0];
  if (url.startsWith("/hub/")) return proxy.emit("request", req, res);
  if (url === "/") { res.writeHead(200, { "Content-Type": "text/html" }); res.end(page); }
  else if (url === "/probe.js") script(res, fs.readFileSync(probe));
  else if (url === "/driver.js") script(res, driver);
  else if (url === "/fetch.js" || url === "/browser-host.js") script(res, fs.readFileSync(path.join(hfHub, url)));
  else if (url === "/input.pt") { res.writeHead(200, { "Content-Type": "application/octet-stream" }); res.end(fs.readFileSync(input)); }
  else if (url.startsWith("/model/") && modelFiles.has(url.slice(7)) && fs.existsSync(path.join(modelDir, url.slice(7)))) {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(fs.readFileSync(path.join(modelDir, url.slice(7))));
  } else { res.writeHead(404); res.end(); }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));

let failures = 0;
const expect = (cond, what) => { console.log((cond ? "ok   " : "FAIL ") + what); if (!cond) failures++; };

const browser = await chromium.launch();
try {
  console.log("engine: Chromium " + browser.version());
  const p = await browser.newPage();
  const errors = [];
  const stdout = [];
  p.on("pageerror", (e) => errors.push(String(e)));
  p.on("console", (m) => {
    // A console message is a chunk of stdout: whole lines.
    stdout.push(...m.text().split("\n"));
  });
  await p.goto(`http://127.0.0.1:${server.address().port}/`);
  const error = await p.evaluate(() => window.runProbe());
  expect(error === null, "download, pin check and inference completed" + (error ? ": " + error : ""));
  expect(errors.length === 0, "no page errors" + (errors.length ? ": " + errors.join("; ") : ""));
  // Blank lines are dropped on both sides: the console has no blank line to
  // show for the "\n" that opens each row of the output dump.
  const lines = (all) => all.filter((l) => l !== "");
  const expected = lines(fs.readFileSync(golden, "utf8").split("\n"));
  const actual = lines(stdout);
  const first = expected.findIndex((l, i) => l !== actual[i]);
  expect(first < 0 && actual.length === expected.length,
    `output equals the native golden (${expected.length} lines)` +
    (first < 0 ? "" : `; first difference at line ${first + 1}:\n  native:  ${expected[first]}\n  browser: ${actual[first]}`));
} finally { await browser.close(); server.close(); }
process.exit(failures ? 1 : 0);
