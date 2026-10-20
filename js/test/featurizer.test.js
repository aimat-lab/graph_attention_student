/**
 * Verifies that the reference JavaScript featurizer reproduces the Python featurization
 * exactly, and that the exported ONNX model reproduces the PyTorch outputs.
 *
 * The fixtures are produced by the Python exporter. Generate them with:
 *
 *     python -m graph_attention_student.cli export --output-path js/test/fixtures
 *
 * or, in the Python test suite, by tests/test_torch_export.py. The test skips itself
 * when the fixture directory is absent, so `npm test` is safe to run on a fresh clone.
 */
import { test, describe, before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

import { Featurizer } from '../featurizer.js';

const require = createRequire(import.meta.url);
const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE_DIR = process.env.MEGAN_FIXTURES || path.join(HERE, 'fixtures');

const available = fs.existsSync(path.join(FIXTURE_DIR, 'fixtures.json'));

describe('featurizer', { skip: available ? false : `no fixtures in ${FIXTURE_DIR}` }, () => {
  let featurizer;
  let fixtures;
  let spec;

  before(async () => {
    const initRDKit = require('@rdkit/rdkit');
    const RDKit = await initRDKit();
    spec = JSON.parse(fs.readFileSync(path.join(FIXTURE_DIR, 'featurization.json'), 'utf8'));
    fixtures = JSON.parse(fs.readFileSync(path.join(FIXTURE_DIR, 'fixtures.json'), 'utf8'));
    featurizer = new Featurizer(RDKit, spec);
  });

  test('reproduces the Python node and edge features exactly', () => {
    assert.ok(fixtures.featurization?.length, 'fixtures carry no featurization section');

    for (const expected of fixtures.featurization) {
      const got = featurizer.process(expected.smiles);

      assert.deepEqual(
        got.nodeAttributes, expected.node_attributes,
        `node_attributes differ for ${expected.smiles}`);
      assert.deepEqual(
        got.edgeIndices, expected.edge_indices,
        `edge_indices differ for ${expected.smiles}`);
      assert.deepEqual(
        got.edgeAttributes, expected.edge_attributes,
        `edge_attributes differ for ${expected.smiles}`);
    }
  });

  test('encodes aromatic bonds as AROMATIC, not as kekulized alternating orders', () => {
    // A regression guard: RDKit.js reports kekulized bond orders in its CommonChem
    // output, so reading `bo` directly would encode benzene as DOUBLE/SINGLE and
    // disagree with the Python featurization on every aromatic molecule.
    const bondEncoder = spec.edgeAttributes.encoders.find((e) => e.property === 'GetBondType');
    if (!bondEncoder) return;

    const aromaticSlot = bondEncoder.values.indexOf(12);
    assert.ok(aromaticSlot >= 0, 'the bond vocabulary has no AROMATIC entry');

    const { edgeAttributes } = featurizer.process('c1ccccc1');
    for (const row of edgeAttributes) {
      assert.equal(
        row[bondEncoder.slice[0] + aromaticSlot], 1,
        'a benzene bond was not encoded as aromatic');
    }
  });

  test('rejects a molecule with no bonds, rather than emitting an empty graph', () => {
    assert.throws(() => featurizer.process('C'), /no bonds/);
  });
});

describe('onnx model', { skip: available ? false : `no fixtures in ${FIXTURE_DIR}` }, () => {
  test('reproduces the PyTorch outputs under ONNX Runtime Web', async () => {
    const ort = (await import('onnxruntime-web')).default;
    ort.env.wasm.numThreads = 1;

    const fx = JSON.parse(fs.readFileSync(path.join(FIXTURE_DIR, 'fixtures.json'), 'utf8'));
    const session = await ort.InferenceSession.create(
      fs.readFileSync(path.join(FIXTURE_DIR, 'model.onnx')),
      { executionProviders: ['wasm'] });

    const feeds = {};
    for (const [name, s] of Object.entries(fx.inputs)) {
      const data = s.type === 'int64'
        ? BigInt64Array.from(s.data.map(BigInt))
        : Float32Array.from(s.data);
      feeds[name] = new ort.Tensor(s.type, data, s.dims);
    }

    const results = await session.run(feeds);

    for (const [name, expected] of Object.entries(fx.expected)) {
      const got = results[name];
      assert.ok(got, `the model produced no output named "${name}"`);
      assert.deepEqual([...got.dims], expected.dims, `dims differ for "${name}"`);

      let worst = 0;
      for (let i = 0; i < expected.data.length; i++) {
        worst = Math.max(worst, Math.abs(got.data[i] - expected.data[i]));
      }
      // Loose enough for float32 reassociation between the two runtimes, tight enough
      // that a genuine wiring error cannot hide under it.
      assert.ok(worst < 1e-4, `"${name}" differs from PyTorch by ${worst}`);
    }
  });
});
