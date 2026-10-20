/**
 * Broad validation: does megan.js featurize thousands of molecules exactly as Python does?
 *
 * Generate the cases first with scripts/validate_js_featurizer.py. This is deliberately
 * not part of `npm test` - it is a bulk check to run when the featurizer changes, while
 * the CI fixtures cover the hand-picked branch cases.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { Featurizer } from '../graph_attention_student/torch/export/megan.js';

const require = createRequire(import.meta.url);
const HERE = path.dirname(fileURLToPath(import.meta.url));
const DIR = path.join(HERE, 'test', 'broad');

const initRDKit = require('@rdkit/rdkit');
const RDKit = await initRDKit();
const spec = JSON.parse(fs.readFileSync(path.join(DIR, 'featurization.json'), 'utf8'));
const cases = JSON.parse(fs.readFileSync(path.join(DIR, 'cases.json'), 'utf8'));
const featurizer = new Featurizer(RDKit, spec);

const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
let ok = 0;
let values = 0;
const failures = [];

for (const c of cases) {
  let got;
  try {
    got = featurizer.process(c.smiles);
  } catch (e) {
    failures.push({ smiles: c.smiles, why: `threw: ${e.message.slice(0, 70)}` });
    continue;
  }
  const nodeOk = eq(got.nodeAttributes, c.node_attributes);
  const idxOk = eq(got.edgeIndices, c.edge_indices);
  const edgeOk = eq(got.edgeAttributes, c.edge_attributes);
  values += c.node_attributes.length * c.node_attributes[0].length
    + c.edge_attributes.length * (c.edge_attributes[0]?.length ?? 0);

  if (nodeOk && idxOk && edgeOk) {
    ok++;
  } else {
    const why = [!nodeOk && 'node', !idxOk && 'edge_indices', !edgeOk && 'edge_attr']
      .filter(Boolean).join('+');
    failures.push({ smiles: c.smiles, why });
  }
}

console.log(`molecules      : ${cases.length}`);
console.log(`exact match    : ${ok} (${(100 * ok / cases.length).toFixed(3)}%)`);
console.log(`feature values : ${values}`);
console.log(`mismatches     : ${failures.length}`);
for (const f of failures.slice(0, 12)) {
  console.log(`   ${f.why.padEnd(24)} ${f.smiles.slice(0, 58)}`);
}
