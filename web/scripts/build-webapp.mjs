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
  ? (await readdir(join(root, 'modules/pytorch.models.pt2/models'))).sort()
  : (process.env.MODELS || 'resnet18').split(/\s+/).filter(Boolean).sort());

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
  await cp(join(root, 'web/node_modules/ai-edge-model-explorer-visualizer/dist'), join(temporary, 'vendor'), { recursive: true });
  await mkdir(join(temporary, 'models'));
const models = [];
  for (const id of ids) {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(id)) throw new Error(`invalid model id: ${id}`);
    const source = join(root, 'modules/pytorch.models.pt2/models', id, 'models/model.json');
    if (!(await stat(source)).isFile()) throw new Error(`unknown model: ${id}`);
    const data = await readFile(source);
    await cp(source, join(temporary, 'models', `${id}.json`));
    models.push({ id, displayName: id, bytes: String(data.byteLength), sha256: createHash('sha256').update(data).digest('hex'), url: `models/${id}.json` });
  }
  if (new Set(ids).size !== ids.length) throw new Error('duplicate model id');
  const { stdout: sourceRef } = await exec('git', ['rev-parse', 'HEAD:modules/pytorch.models.pt2'], { cwd: root });
  await writeFile(join(temporary, 'catalog.json'), `${JSON.stringify({ schemaVersion: 1, sourceRef: sourceRef.trim(), defaultModel: models[0].id, models })}\n`);
  await assertArtifact(temporary);
  await rm(output, { recursive: true, force: true });
  await rename(temporary, output);
} catch (error) { await rm(temporary, { recursive: true, force: true }); throw error; }
