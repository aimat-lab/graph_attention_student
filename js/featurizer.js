/**
 * Reference JavaScript featurizer for MEGAN ONNX models.
 *
 * The exported ONNX graph takes graph tensors, not SMILES, so a consumer has to build
 * `x` and `edge_attr` itself - column for column, in exactly the order the model was
 * trained on. That order is described by the `featurization.json` written next to the
 * model, and this module is the reference implementation of it.
 *
 * Chemistry comes from RDKit.js, but only through its portable surface: `mol.get_json()`,
 * which yields atomic numbers, formal charges, hydrogen counts, radicals, isotopes,
 * kekulized bond orders, and an extension block listing aromatic atoms, aromatic bonds
 * and rings. RDKit.js exposes no per-atom API, so several properties the featurizer needs
 * are recomputed from that data by the ports in `rdkit_algorithms.js`.
 *
 * Usage:
 *
 *     import initRDKit from '@rdkit/rdkit';
 *     import { Featurizer } from './featurizer.js';
 *
 *     const RDKit = await initRDKit();
 *     const spec = await (await fetch('./featurization.json')).json();
 *     const manifest = await (await fetch('./manifest.json')).json();
 *     const featurizer = new Featurizer(RDKit, spec);
 *     const graph = featurizer.process('CCO');
 *     const feeds = Featurizer.toFeeds(
 *       [graph], ort, manifest.architecture.config.num_channels);
 */

import { ATOMIC_MASS, ELEMENT_SYMBOL, ISOTOPE_MASS } from './periodic_table.js';
import { setConjugation, hybridizations, topology } from './rdkit_algorithms.js';

/** RDKit BondType integers, as the one-hot vocabularies use them. */
const BOND_TYPE = { SINGLE: 1, DOUBLE: 2, TRIPLE: 3, AROMATIC: 12 };

/**
 * RDKit HybridizationType integers.
 *
 * Note SP2D (square planar) sits at 5, BETWEEN SP3 and SP3D - so the sequence is not the
 * obvious one. Assuming SP3D = 5 silently shifts every five- and six-coordinate atom by
 * one one-hot slot.
 */
const HYBRIDIZATION = {
  UNSPECIFIED: 0, S: 1, SP: 2, SP2: 3, SP3: 4, SP2D: 5, SP3D: 6, SP3D2: 7, OTHER: 8,
};

/**
 * RDKit BondStereo integers, keyed by the string CommonChem reports.
 * Absent means STEREONONE.
 */
const BOND_STEREO = { unspecified: 0, any: 1, cis: 2, trans: 3 };

/** Everything this featurizer knows how to compute. */
export const SUPPORTED_ATOM_PROPERTIES = new Set([
  'GetSymbol', 'GetFormalCharge', 'GetIsAromatic', 'GetDegree',
  'GetTotalDegree', 'GetTotalNumHs', 'GetMass', 'IsInRing', 'GetHybridization',
]);

export const SUPPORTED_BOND_PROPERTIES = new Set([
  'GetBondType', 'GetIsAromatic', 'IsInRing', 'GetStereo', 'GetIsConjugated',
]);

export class UnsupportedSpecError extends Error {}

export class Featurizer {
  /**
   * @param {object} RDKit an initialised RDKit.js module
   * @param {object} spec the parsed `featurization.json`
   */
  constructor(RDKit, spec) {
    this.RDKit = RDKit;
    this.spec = spec;

    for (const [encoders, supported, kind] of [
      [spec.nodeAttributes.encoders, SUPPORTED_ATOM_PROPERTIES, 'atom'],
      [spec.edgeAttributes.encoders, SUPPORTED_BOND_PROPERTIES, 'bond'],
    ]) {
      for (const encoder of encoders) {
        if (!supported.has(encoder.property)) {
          throw new UnsupportedSpecError(
            `featurization.json asks for the ${kind} property "${encoder.property}", ` +
            `which this featurizer does not implement. Supported: ` +
            `${[...supported].join(', ')}.`);
        }
      }
    }
  }

  /**
   * One-hot encode per a `kind: 'one_hot'` encoder description.
   *
   * Mirrors visual_graph_datasets' OneHotEncoder, including the two behaviours that are
   * easy to get wrong: with `add_unknown` there is a trailing slot set when nothing
   * matched, and WITHOUT it an unmatched value yields an all-zero row rather than an
   * error.
   */
  static oneHot(value, encoder) {
    const width = encoder.values.length + (encoder.add_unknown ? 1 : 0);
    const out = new Array(width).fill(0);
    const index = encoder.values.indexOf(value);
    if (index >= 0) {
      out[index] = 1;
    } else if (encoder.add_unknown) {
      out[width - 1] = 1;
    }
    return out;
  }

  /**
   * Parse a molecule into the plain atom/bond arrays everything downstream works from.
   */
  parse(smiles) {
    const mol = this.RDKit.get_mol(smiles);
    if (!mol) throw new Error(`RDKit could not parse the SMILES "${smiles}"`);

    try {
      const parsed = JSON.parse(mol.get_json());
      const chem = parsed.molecules[0];
      const atomDefaults = parsed.defaults?.atom ?? {};
      const bondDefaults = parsed.defaults?.bond ?? {};

      const rdkit = (chem.extensions ?? [])
        .find((e) => e.name === 'rdkitRepresentation') ?? {};
      const aromaticAtoms = new Set(rdkit.aromaticAtoms ?? []);
      const aromaticBonds = new Set(rdkit.aromaticBonds ?? []);
      const rings = rdkit.atomRings ?? [];

      const atoms = chem.atoms.map((raw, index) => {
        const atom = { ...atomDefaults, ...raw };
        return {
          z: atom.z,
          chg: atom.chg || 0,
          impHs: atom.impHs || 0,
          nRad: atom.nRad || 0,
          isotope: atom.isotope || 0,
          aromatic: aromaticAtoms.has(index),
          // 'cw' / 'ccw' -> RDKit's CHI_TETRAHEDRAL_CW / _CCW, which short-circuit
          // hybridization to SP3 for a four-substituent stereocentre.
          chiral: atom.stereo === 'cw' || atom.stereo === 'ccw' ? atom.stereo : null,
        };
      });

      // CommonChem bond orders are KEKULIZED - benzene arrives as alternating bo:2 / bo:1,
      // never bo:12 - whereas Python's Chem.MolFromSmiles does not kekulize, so
      // GetBondType() reports AROMATIC there. Taking `bo` at face value would disagree
      // with the training featurization on every aromatic ring. RDKit's own extension
      // block records which bonds are aromatic, and that is what reconciles the two.
      const bonds = chem.bonds.map((raw, index) => {
        const bond = { ...bondDefaults, ...raw };
        return {
          i: bond.atoms[0],
          j: bond.atoms[1],
          order: typeof bond.bo === 'number' ? bond.bo : 1,
          aromatic: aromaticBonds.has(index),
          stereo: bond.stereo ?? 'unspecified',
        };
      });

      // Ring membership: an atom is in a ring if any ring lists it; a bond is in a ring
      // if its two atoms are adjacent within one.
      const inRingAtom = atoms.map(() => false);
      const ringEdges = new Set();
      for (const ring of rings) {
        for (let k = 0; k < ring.length; k++) {
          inRingAtom[ring[k]] = true;
          const a = ring[k];
          const b = ring[(k + 1) % ring.length];
          ringEdges.add(a < b ? `${a}-${b}` : `${b}-${a}`);
        }
      }
      const inRingBond = bonds.map(({ i, j }) =>
        ringEdges.has(i < j ? `${i}-${j}` : `${j}-${i}`));

      const { degree } = topology(atoms, bonds);

      return {
        atoms,
        bonds,
        inRingAtom,
        inRingBond,
        degree,
        conjugated: setConjugation(atoms, bonds),
        hybridization: hybridizations(atoms, bonds),
      };
    } finally {
      mol.delete();
    }
  }

  /** @returns {number} the value of one atom property, in RDKit's own encoding. */
  atomProperty(ctx, index, property) {
    const atom = ctx.atoms[index];
    switch (property) {
      // CommonChem carries the atomic number, never a symbol, and omits carbon and a
      // zero charge entirely via the "defaults" block merged in during parsing.
      case 'GetSymbol': return ELEMENT_SYMBOL[atom.z] ?? `Z${atom.z}`;
      case 'GetFormalCharge': return atom.chg;
      case 'GetIsAromatic': return atom.aromatic;
      case 'GetDegree': return ctx.degree[index];
      case 'GetTotalDegree': return ctx.degree[index] + atom.impHs;
      case 'GetTotalNumHs': return atom.impHs;
      case 'GetMass': {
        if (!atom.isotope) return ATOMIC_MASS[atom.z] ?? 0;
        // An isotope-labelled atom carries its exact mass, not its mass number.
        return ISOTOPE_MASS[`${atom.z}_${atom.isotope}`] ?? atom.isotope;
      }
      case 'IsInRing': return ctx.inRingAtom[index];
      case 'GetHybridization': return HYBRIDIZATION[ctx.hybridization[index]] ?? 0;
      default: throw new Error(`unsupported atom property "${property}"`);
    }
  }

  /** @returns {number} the value of one bond property, in RDKit's own encoding. */
  bondProperty(ctx, index, property) {
    const bond = ctx.bonds[index];
    switch (property) {
      // The aromatic flag wins over the kekulized order - see the note in parse().
      case 'GetBondType': return bond.aromatic ? BOND_TYPE.AROMATIC : bond.order;
      case 'GetIsAromatic': return bond.aromatic;
      case 'IsInRing': return ctx.inRingBond[index];
      case 'GetStereo': return BOND_STEREO[bond.stereo] ?? 0;
      case 'GetIsConjugated': return ctx.conjugated[index];
      default: throw new Error(`unsupported bond property "${property}"`);
    }
  }

  /** Build one feature row from a list of encoder descriptions. */
  row(encoders, read) {
    const out = [];
    for (const encoder of encoders) {
      const value = read(encoder.property);
      if (encoder.kind === 'one_hot') {
        out.push(...Featurizer.oneHot(value, encoder));
      } else {
        // Booleans reach the model as 1.0 / 0.0, matching numpy's coercion in Python.
        out.push(typeof value === 'boolean' ? (value ? 1 : 0) : value);
      }
    }
    return out;
  }

  /**
   * Turn a SMILES string into the graph arrays the ONNX model consumes.
   *
   * Node order is the RDKit atom index order, and every bond contributes two edges -
   * forward then reverse, back to back - which is what `MoleculeProcessing.process`
   * does with `double_edges_undirected=True`.
   *
   * @returns {{nodeAttributes: number[][], edgeIndices: number[][], edgeAttributes: number[][]}}
   */
  process(smiles) {
    const ctx = this.parse(smiles);

    if (ctx.bonds.length === 0) {
      throw new Error(
        `"${smiles}" has no bonds. MEGAN cannot process an edgeless graph; this is a ` +
        `limitation of the model itself, not of the featurizer.`);
    }

    const nodeAttributes = ctx.atoms.map((_, index) =>
      this.row(this.spec.nodeAttributes.encoders,
        (property) => this.atomProperty(ctx, index, property)));

    const edgeIndices = [];
    const edgeAttributes = [];
    ctx.bonds.forEach((bond, index) => {
      const features = this.row(this.spec.edgeAttributes.encoders,
        (property) => this.bondProperty(ctx, index, property));
      edgeIndices.push([bond.i, bond.j]);
      edgeAttributes.push(features);
      edgeIndices.push([bond.j, bond.i]);
      edgeAttributes.push(features);
    });

    return { nodeAttributes, edgeIndices, edgeAttributes };
  }

  /**
   * Collate graphs into the ONNX feed dict, mirroring PyG's batching: node and edge
   * arrays are concatenated and edge indices offset by the running node count.
   *
   * @param {Array} graphs output of `process`
   * @param {object} ort the onnxruntime module, for its Tensor constructor
   * @param {number} numChannels the model's `num_channels`, for the node_mask. Read it
   *   from `manifest.architecture.config.num_channels`; there is deliberately no
   *   default, since a wrong guess only fails once the session runs.
   */
  static toFeeds(graphs, ort, numChannels) {
    if (!Number.isInteger(numChannels) || numChannels < 1) {
      throw new Error(
        `toFeeds needs the model's number of explanation channels, got ${numChannels}. ` +
        `Read it from manifest.json as manifest.architecture.config.num_channels.`);
    }

    const x = [];
    const batch = [];
    const src = [];
    const dst = [];
    const edgeAttr = [];
    let offset = 0;

    graphs.forEach((graph, graphIndex) => {
      for (const row of graph.nodeAttributes) {
        x.push(...row);
        batch.push(graphIndex);
      }
      graph.edgeIndices.forEach(([i, j], k) => {
        src.push(i + offset);
        dst.push(j + offset);
        edgeAttr.push(...graph.edgeAttributes[k]);
      });
      offset += graph.nodeAttributes.length;
    });

    const numNodes = batch.length;
    const numEdges = src.length;
    const nodeSize = graphs[0].nodeAttributes[0].length;
    const edgeSize = graphs[0].edgeAttributes[0].length;
    const big = (values) => BigInt64Array.from(values.map((v) => BigInt(v)));

    return {
      x: new ort.Tensor('float32', Float32Array.from(x), [numNodes, nodeSize]),
      // edge_index is (2, E): all sources, then all targets.
      edge_index: new ort.Tensor('int64', big([...src, ...dst]), [2, numEdges]),
      edge_attr: new ort.Tensor('float32', Float32Array.from(edgeAttr), [numEdges, edgeSize]),
      batch: new ort.Tensor('int64', big(batch), [numNodes]),
      graph_ids: new ort.Tensor('int64', big(graphs.map((_, i) => i)), [graphs.length]),
      node_mask: new ort.Tensor(
        'float32', Float32Array.from(new Array(numNodes * numChannels).fill(1)),
        [numNodes, numChannels]),
    };
  }
}
