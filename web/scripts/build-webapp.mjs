import { cp, lstat, mkdir, mkdtemp, readdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const exec = promisify(execFile);
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const temporary = await mkdtemp(join(root, '.webapp-dist-'));
const output = join(root, 'webapp-dist');
const ids = (process.env.MODELS === 'all'
  ? (await readdir(join(root, 'modules/devcontainer.pytorch-image-models/models'))).sort()
  : (process.env.MODELS || 'mobilenetv2_050').split(/\s+/).filter(Boolean).sort());

/* The payload-free sweep (`.ai/pt2_model_support.md`) keyed by model id, so the
 * catalogue can carry each model's known Native4D/Kernel capability alongside
 * it. A model absent from the sweep -- stale relative to the submodule pin, or
 * simply never swept -- carries `support: null`; the page must never filter or
 * disable on an absent answer, only on one the sweep actually gives. */
async function readSupport(root) {
  const text = await readFile(join(root, 'test/data/pt2_json_model_support.jsonl'), 'utf8');
  const byId = new Map();
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    byId.set(row.model, row);
  }
  return byId;
}

function catalogSupport(row) {
  if (!row) return null;
  return {
    nativeBuilds: row.native_builds === true,
    native4dConverts: row.native4d_converts,
    native4dBlocker: row.native4d_blocker,
    kernelConverts: row.kernel_converts,
    kernelBlocker: row.kernel_blocker,
  };
}

async function assertArtifact(directory, relative = '') {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    const child = relative ? `${relative}/${entry.name}` : entry.name;
    const info = await lstat(path);
    if (info.isSymbolicLink()) throw new Error(`artifact contains symlink: ${child}`);
    if (info.isDirectory()) { await assertArtifact(path, child); continue; }
    if (!info.isFile()) throw new Error(`artifact contains non-file: ${child}`);
    if (child.startsWith('models/') && child.endsWith('.json')) continue;
    if (child.startsWith('vendor/')) continue;
    if (new Set(['index.html', 'app.css', 'app.js', 'coordinator.js', 'panels.js', 'presentation.js', 'renderer.js', 'source_store.js', 'worker.js', 'webapp_bridge.js', 'catalog.json']).has(child)) continue;
    throw new Error(`artifact contains unexpected file: ${child}`);
  }
}

try {
  await cp(join(root, 'web/app'), temporary, { recursive: true });
  await cp(join(root, '_build/default/js/webapp/webapp_worker.bc.js'), join(temporary, 'worker.js'));
  await cp(join(root, '_build/default/js/webapp/webapp_bridge.bc.js'), join(temporary, 'webapp_bridge.js'));
  // Built from the submodule that pins the OCaml schema, not from npm -- see
  // the `visualizer.build` block in the Makefile for why there is no published
  // release that carries what `lib/model_explorer_export` emits. Checked
  // explicitly because `cp`'s ENOENT names a path but not the target that
  // produces it.
  const visualizer = join(root, 'vendored/ocaml-model-explorer/model-explorer/src/ui/custom_element_npm/dist');
  if (!(await stat(join(visualizer, 'main_browser.js')).catch(() => null))?.isFile()) {
    throw new Error(`no visualizer bundle at ${visualizer}; run \`make visualizer.build\``);
  }
  await cp(visualizer, join(temporary, 'vendor'), { recursive: true });
  await mkdir(join(temporary, 'models'));
  const support = await readSupport(root);
const models = [];
  for (const id of ids) {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(id)) throw new Error(`invalid model id: ${id}`);
    const source = join(root, 'modules/devcontainer.pytorch-image-models/models', id, 'models/model.json');
    if (!(await stat(source)).isFile()) throw new Error(`unknown model: ${id}`);
    const data = await readFile(source);
    await cp(source, join(temporary, 'models', `${id}.json`));
    models.push({
      id, displayName: id, bytes: String(data.byteLength), sha256: createHash('sha256').update(data).digest('hex'),
      url: `models/${id}.json`, support: catalogSupport(support.get(id)),
    });
  }
  if (new Set(ids).size !== ids.length) throw new Error('duplicate model id');
  const { stdout: sourceRef } = await exec('git', ['rev-parse', 'HEAD:modules/devcontainer.pytorch-image-models'], { cwd: root });
  await writeFile(join(temporary, 'catalog.json'), `${JSON.stringify({ schemaVersion: 1, sourceRef: sourceRef.trim(), defaultModel: models[0].id, models })}\n`);
  await assertArtifact(temporary);
  await rm(output, { recursive: true, force: true });
  await rename(temporary, output);
} catch (error) { await rm(temporary, { recursive: true, force: true }); throw error; }
