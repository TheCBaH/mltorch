// Runs safetensors_probe.bc.js under node, with the checkpoint downloaded by
// hf-hub's node host into the huggingface_hub cache (the layout the native
// driver shares, so a warm cache needs no network).
//
// usage: node node_run.cjs <safetensors_probe.bc.js> <model_dir> <input.pt>
// HF_HOME, HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE, XDG_CACHE_HOME and HF_ENDPOINT
// are read as hf-hub's own node.cjs reads them.
'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const [probe, modelDir, input] = process.argv.slice(2);
if (!probe || !modelDir || !input) {
  console.error('usage: node_run.cjs <safetensors_probe.bc.js> <model_dir> <input.pt>');
  process.exit(2);
}

const host = require('../../vendored/ocaml-hf-hub/javascript/node-host.cjs');
globalThis.hfHubHost = { call: host.call, bytes: (file) => fs.readFileSync(file) };
require(path.resolve(probe));

const env = process.env;
const home = env.HF_HOME || path.join(env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache'), 'huggingface');
const read = (rel) => fs.readFileSync(path.join(modelDir, rel), 'utf8');
const constants = path.join('data', 'constants', 'model_constants_config.json');

globalThis.safetensorsProbe({
  program: read('models/model.json'),
  weights: read('data/weights/model_weights_config.json'),
  constants: fs.existsSync(path.join(modelDir, constants)) ? read(constants) : null,
  map: read('models/safetensors.json'),
  input: fs.readFileSync(input),
  cacheDir: env.HF_HUB_CACHE || env.HUGGINGFACE_HUB_CACHE || path.join(home, 'hub'),
  endpoint: env.HF_ENDPOINT || 'https://huggingface.co',
}, (error) => {
  if (error !== null) { console.error('safetensors_probe: ' + error); process.exitCode = 1; }
});
