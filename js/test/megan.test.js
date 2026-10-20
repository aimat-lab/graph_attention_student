/**
 * Verifies megan.js: that the featurizer reproduces the Python featurization exactly, that
 * the exported ONNX model reproduces the PyTorch outputs, and that MeganModel hands those
 * outputs back molecule by molecule.
 *
 * The fixtures are produced by the Python exporter. Generate them with:
 *
 *     python scripts/generate_js_fixtures.py
 *
 * or point MEGAN_FIXTURES at any export folder. The tests that need fixtures skip
 * themselves when the folder is absent, so `npm test` is safe to run on a fresh clone.
 */
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

import {
  Featurizer, MeganModel, UnsupportedSpecError,
} from '../../graph_attention_student/torch/export/megan.js';

const require = createRequire(import.meta.url);
const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE_DIR = process.env.MEGAN_FIXTURES || path.join(HERE, 'fixtures');

const available = fs.existsSync(path.join(FIXTURE_DIR, 'fixtures.json'));
const skip = available ? false : `no fixtures in ${FIXTURE_DIR}`;
const read = (name) => JSON.parse(fs.readFileSync(path.join(FIXTURE_DIR, name), 'utf8'));

const RDKit = await require('@rdkit/rdkit')();
const ort = (await import('onnxruntime-web')).default;
ort.env.wasm.numThreads = 1;

const createSession = () => ort.InferenceSession.create(
  fs.readFileSync(path.join(FIXTURE_DIR, 'model.onnx')), { executionProviders: ['wasm'] });

/** A minimal contract, for the tests that need a featurizer but no fixtures. */
const TINY_SPEC = {
  specVersion: 1,
  generatedBy: 'graph_attention_student test',
  nodeAttributes: {
    size: 3,
    encoders: [{ property: 'GetSymbol', kind: 'one_hot', values: ['C', 'O'], add_unknown: true }],
  },
  edgeAttributes: { size: 1, encoders: [{ property: 'IsInRing', kind: 'identity' }] },
};

const tinyManifest = (overrides = {}) => ({
  schemaVersion: 1,
  generatedBy: 'graph_attention_student test',
  model: { path: 'model.onnx' },
  architecture: { config: { num_channels: 2, prediction_mode: 'regression' } },
  featurization: TINY_SPEC,
  ...overrides,
});

/**
 * Stands in for a session, returning outputs whose values are their own flat positions,
 * so that a test can tell exactly which rows ended up with which molecule.
 */
const stubSession = (graphOutput) => ({
  async run(feeds) {
    const [numNodes] = feeds.x.dims;
    const [numEdges] = feeds.edge_attr.dims;
    const [numGraphs] = feeds.graph_ids.dims;
    const positions = (n) => Float32Array.from({ length: n }, (_, i) => i);
    return {
      graph_output: new ort.Tensor(
        'float32', Float32Array.from(graphOutput), [numGraphs, graphOutput.length / numGraphs]),
      node_importance_norm: new ort.Tensor('float32', positions(numNodes * 2), [numNodes, 2]),
      edge_importance_norm: new ort.Tensor('float32', positions(numEdges * 2), [numEdges, 2]),
    };
  },
});

const close = (got, expected) => got.every((value, i) => Math.abs(value - expected[i]) < 1e-6);

describe('featurizer', { skip }, () => {
  test('reproduces the Python node and edge features exactly', () => {
    const fixtures = read('fixtures.json');
    const featurizer = new Featurizer(RDKit, read('featurization.json'));
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
    const spec = read('featurization.json');
    const featurizer = new Featurizer(RDKit, spec);
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
    const featurizer = new Featurizer(RDKit, read('featurization.json'));
    assert.throws(() => featurizer.process('C'), /no bonds/);
  });
});

describe('featurization contract', () => {
  test('refuses a specVersion it does not read, naming the version that wrote it', () => {
    const spec = { ...TINY_SPEC, specVersion: 2, generatedBy: 'graph_attention_student 9.9.0' };
    assert.throws(() => new Featurizer(RDKit, spec), (error) =>
      error instanceof UnsupportedSpecError
      && /specVersion 2/.test(error.message)
      && /9\.9\.0/.test(error.message)
      && /same export folder/.test(error.message));
  });

  test('refuses a property it does not implement', () => {
    const spec = structuredClone(TINY_SPEC);
    spec.nodeAttributes.encoders.push({ property: 'GetChiralTag', kind: 'identity' });
    assert.throws(() => new Featurizer(RDKit, spec), /"GetChiralTag"/);
  });
});

describe('toFeeds', { skip }, () => {
  test('collates featurized molecules into exactly the tensors PyG built', () => {
    // The fixture inputs are the PyG batch the model was traced with, so this covers the
    // whole consumer path up to the session: featurization, edge offsets, batch vector,
    // graph_ids and node_mask.
    const fixtures = read('fixtures.json');
    const manifest = read('manifest.json');

    const featurizer = new Featurizer(RDKit, read('featurization.json'));
    const graphs = fixtures.featurization.map((entry) => featurizer.process(entry.smiles));
    const feeds = Featurizer.toFeeds(graphs, ort, manifest.architecture.config.num_channels);

    for (const [name, expected] of Object.entries(fixtures.inputs)) {
      const got = feeds[name];
      assert.ok(got, `toFeeds produced no input named "${name}"`);
      assert.deepEqual([...got.dims], expected.dims, `dims differ for "${name}"`);
      assert.deepEqual(Array.from(got.data, Number), expected.data, `values differ for "${name}"`);
    }
  });

  test('refuses to guess the number of channels', () => {
    const graph = {
      nodeAttributes: [[1], [1]], edgeIndices: [[0, 1], [1, 0]], edgeAttributes: [[1], [1]],
    };
    assert.throws(() => Featurizer.toFeeds([graph], ort), /num_channels/);
  });
});

describe('onnx model', { skip }, () => {
  test('reproduces the PyTorch outputs under ONNX Runtime Web', async () => {
    const fx = read('fixtures.json');
    const session = await createSession();

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

describe('MeganModel', () => {
  test('hands back the PyTorch outputs, split per molecule', { skip }, async () => {
    const fixtures = read('fixtures.json');
    const model = new MeganModel({
      manifest: read('manifest.json'), session: await createSession(), RDKit, ort,
    });
    const results = await model.predict(fixtures.featurization.map((entry) => entry.smiles));

    results.forEach((result, i) => {
      const molecule = fixtures.featurization[i];
      assert.equal(result.smiles, molecule.smiles);
      assert.equal(result.nodeImportance.length, molecule.node_attributes.length);
      assert.equal(result.edgeImportance.length, molecule.edge_indices.length);
    });

    // Put back together in order, the molecules' outputs are the batch PyTorch computed.
    for (const [name, expected] of Object.entries(fixtures.expected)) {
      const got = results.flatMap((result) => [result.outputs[name]].flat(Infinity));
      assert.equal(got.length, expected.data.length, `"${name}" has the wrong size`);
      const worst = Math.max(...got.map((value, i) => Math.abs(value - expected.data[i])));
      assert.ok(worst < 1e-4, `"${name}" differs from PyTorch by ${worst}`);
    }
  });

  test('loads an export folder over HTTP', { skip }, async () => {
    const parent = path.dirname(FIXTURE_DIR);
    const server = http.createServer((request, response) => {
      const file = path.join(parent, decodeURIComponent(new URL(request.url, 'http://x').pathname));
      fs.readFile(file, (error, data) => {
        response.writeHead(error ? 404 : 200);
        response.end(error ? undefined : data);
      });
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));

    try {
      // No trailing slash on purpose: load() has to treat the URL as a folder.
      const url = `http://127.0.0.1:${server.address().port}/${path.basename(FIXTURE_DIR)}`;
      const model = await MeganModel.load(url, {
        RDKit, ort, sessionOptions: { executionProviders: ['wasm'] },
      });

      const molecule = read('fixtures.json').featurization[0];
      const [result] = await model.predict([molecule.smiles]);
      assert.equal(result.nodeImportance.length, molecule.node_attributes.length);
    } finally {
      server.closeAllConnections();
      server.close();
    }
  });

  test('splits a batch into molecules at the right rows', async () => {
    const model = new MeganModel({
      manifest: tinyManifest(), session: stubSession([0.5, -1.5]), RDKit, ort,
    });
    // CCO has 3 atoms and 2 bonds, so 4 edges; CO has 2 atoms and 1 bond, so 2 edges.
    const [ethanol, methanol] = await model.predict(['CCO', 'CO']);

    assert.deepEqual(ethanol.nodeImportance, [[0, 1], [2, 3], [4, 5]]);
    assert.deepEqual(methanol.nodeImportance, [[6, 7], [8, 9]]);
    assert.deepEqual(ethanol.edgeImportance, [[0, 1], [2, 3], [4, 5], [6, 7]]);
    assert.deepEqual(methanol.edgeImportance, [[8, 9], [10, 11]]);
    assert.deepEqual(ethanol.graph.edgeIndices, [[0, 1], [1, 0], [1, 2], [2, 1]]);
    // Regression values come back as they are.
    assert.deepEqual(ethanol.prediction, [0.5]);
    assert.deepEqual(methanol.prediction, [-1.5]);
  });

  test('turns classification logits into probabilities', async () => {
    const manifest = tinyManifest({
      architecture: { config: { num_channels: 2, prediction_mode: 'classification' } },
    });
    const model = new MeganModel({
      manifest, session: stubSession([1, 2, 3, 0, 0, 0]), RDKit, ort,
    });
    const [first, second] = await model.predict(['CCO', 'CO']);

    assert.ok(close(first.prediction, [0.09003057, 0.24472847, 0.66524096]));
    assert.ok(close(second.prediction, [1 / 3, 1 / 3, 1 / 3]));
    assert.deepEqual(first.outputs.graph_output, [1, 2, 3], 'the logits are kept as well');
  });

  test('turns multi-label logits into one sigmoid per target', async () => {
    const manifest = tinyManifest({
      architecture: { config: { num_channels: 2, prediction_mode: 'bce' } },
    });
    const model = new MeganModel({ manifest, session: stubSession([0, 2]), RDKit, ort });
    const [result] = await model.predict(['CCO']);

    assert.ok(close(result.prediction, [0.5, 0.88079708]));
  });

  test('refuses a manifest schema it does not read', () => {
    const manifest = tinyManifest({ schemaVersion: 2, generatedBy: 'graph_attention_student 9.9.0' });
    assert.throws(
      () => new MeganModel({ manifest, session: stubSession([0]), RDKit, ort }),
      /schemaVersion 2.*9\.9\.0/s);
  });

  test('says why an export without a featurization contract cannot run', () => {
    const manifest = tinyManifest({
      featurization: null,
      featurizationOmittedBecause: 'encoder "crippen" does not read a plain RDKit property',
    });
    assert.throws(
      () => new MeganModel({ manifest, session: stubSession([0]), RDKit, ort }),
      /no featurization contract.*crippen/s);
  });

  test('takes an array of SMILES, not a single string', async () => {
    const model = new MeganModel({
      manifest: tinyManifest(), session: stubSession([0]), RDKit, ort,
    });
    await assert.rejects(model.predict('CCO'), /array of SMILES/);
  });

  test('needs an absolute URL outside a browser', async () => {
    await assert.rejects(MeganModel.load('onnx_export/', { RDKit, ort }), /not an absolute URL/);
  });
});
