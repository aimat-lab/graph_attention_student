/**
 * megan.js - runs an exported MEGAN model in the browser or in Node.
 *
 * An ONNX export takes graph tensors, not SMILES, so a consumer has to build `x` and
 * `edge_attr` itself - column for column, in exactly the order the model was trained on.
 * That order is described by the export's `featurization.json`, and this module is the
 * reference implementation of it, together with the batching and output handling around
 * the ONNX session.
 *
 * It is one ES module with no dependencies of its own: callers pass in an initialised
 * RDKit.js module and the onnxruntime-web (or onnxruntime-node) module. Every export with a
 * `featurization.json` carries a copy of this file, written by the same package version,
 * so the copy next to a model always matches it.
 *
 * Usage, on a page served next to the export folder:
 *
 *     <script src="https://cdn.jsdelivr.net/npm/@rdkit/rdkit@2025.3.2-1.0.0/dist/RDKit_minimal.js"></script>
 *     <script type="module">
 *       import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.min.mjs';
 *       import { MeganModel } from './onnx_export/megan.js';
 *
 *       const RDKit = await window.initRDKitModule();
 *       const model = await MeganModel.load('onnx_export/', { RDKit, ort });
 *       const [aspirin] = await model.predict(['CC(=O)Oc1ccccc1C(=O)O']);
 *       aspirin.prediction;      // one value per target; probabilities for a classifier
 *       aspirin.nodeImportance;  // one row per atom, one column per explanation channel
 *     </script>
 *
 * `MeganModel` combines two steps that are exported as well, for callers that run the
 * session themselves: `Featurizer.process` turns a SMILES into graph arrays, and
 * `Featurizer.toFeeds` batches graphs into the input tensors exactly as PyTorch Geometric
 * does.
 *
 * Chemistry comes from RDKit.js, but only through its portable surface: `mol.get_json()`,
 * which yields atomic numbers, formal charges, hydrogen counts, radicals, isotopes,
 * kekulized bond orders, and an extension block listing aromatic atoms, aromatic bonds
 * and rings. RDKit.js exposes no per-atom API, so several properties the featurizer needs
 * are recomputed from that data by the ports of RDKit's own algorithms below.
 */

// == Periodic table ==
//
// Constants from RDKit's own periodic table - in Python, Chem.GetPeriodicTable() and its
// GetDefaultValence, GetNOuterElecs, GetValenceList, GetAtomicWeight and GetElementSymbol,
// plus GetMassForIsotope for the isotopes listed. RDKit's conjugation and hybridization
// algorithms are defined in terms of these, so they have to match exactly rather than
// come from a textbook.

const DEFAULT_VALENCE = {1: 1, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 3, 16: 2, 17: 1, 18: 0, 19: 1, 20: 2, 21: -1, 22: -1, 23: -1, 24: -1, 25: -1, 26: -1, 27: -1, 28: -1, 29: -1, 30: -1, 31: 3, 32: 4, 33: 3, 34: 2, 35: 1, 36: 0, 37: 1, 38: 2, 39: -1, 40: -1, 41: -1, 42: -1, 43: -1, 44: -1, 45: -1, 46: -1, 47: -1, 48: -1, 49: 3, 50: 2, 51: 3, 52: 2, 53: 1, 54: 0, 55: 1, 56: 2, 57: -1, 58: -1, 59: -1, 60: -1, 61: -1, 62: -1, 63: -1, 64: -1, 65: -1, 66: -1, 67: -1, 68: -1, 69: -1, 70: -1, 71: -1, 72: -1, 73: -1, 74: -1, 75: -1, 76: -1, 77: -1, 78: -1, 79: -1, 80: -1, 81: -1, 82: 2, 83: 3, 84: 2, 85: 1, 86: 0, 87: 1, 88: 2, 89: -1, 90: -1, 91: -1, 92: -1, 93: -1, 94: -1, 95: -1, 96: -1, 97: -1, 98: -1, 99: -1, 100: -1, 101: -1, 102: -1, 103: -1};

const N_OUTER_ELECS = {1: 1, 2: 2, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 1, 20: 2, 21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10, 29: 11, 30: 2, 31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8, 37: 1, 38: 2, 39: 3, 40: 4, 41: 5, 42: 6, 43: 7, 44: 8, 45: 9, 46: 10, 47: 11, 48: 2, 49: 3, 50: 4, 51: 5, 52: 6, 53: 7, 54: 8, 55: 1, 56: 2, 57: 3, 58: 4, 59: 3, 60: 4, 61: 5, 62: 6, 63: 7, 64: 8, 65: 9, 66: 10, 67: 11, 68: 12, 69: 13, 70: 14, 71: 15, 72: 4, 73: 5, 74: 6, 75: 7, 76: 8, 77: 9, 78: 10, 79: 11, 80: 2, 81: 3, 82: 4, 83: 5, 84: 6, 85: 7, 86: 8, 87: 1, 88: 2, 89: 3, 90: 4, 91: 3, 92: 4, 93: 5, 94: 6, 95: 7, 96: 8, 97: 9, 98: 10, 99: 11, 100: 12, 101: 13, 102: 14, 103: 15};

const VALENCE_LIST = {1: [1], 2: [0], 3: [1,-1], 4: [2], 5: [3], 6: [4], 7: [3], 8: [2], 9: [1], 10: [0], 11: [1,-1], 12: [2,-1], 13: [3], 14: [4], 15: [3,5], 16: [2,4,6], 17: [1], 18: [0], 19: [1,-1], 20: [2,-1], 21: [-1], 22: [-1], 23: [-1], 24: [-1], 25: [-1], 26: [-1], 27: [-1], 28: [-1], 29: [-1], 30: [-1], 31: [3], 32: [4], 33: [3,5], 34: [2,4,6], 35: [1], 36: [0], 37: [1,-1], 38: [2,-1], 39: [-1], 40: [-1], 41: [-1], 42: [-1], 43: [-1], 44: [-1], 45: [-1], 46: [-1], 47: [-1], 48: [-1], 49: [3], 50: [2,4], 51: [3,5], 52: [2,4,6], 53: [1,3,5], 54: [0,2,4,6], 55: [1], 56: [2,-1], 57: [-1], 58: [-1], 59: [-1], 60: [-1], 61: [-1], 62: [-1], 63: [-1], 64: [-1], 65: [-1], 66: [-1], 67: [-1], 68: [-1], 69: [-1], 70: [-1], 71: [-1], 72: [-1], 73: [-1], 74: [-1], 75: [-1], 76: [-1], 77: [-1], 78: [-1], 79: [-1], 80: [-1], 81: [-1], 82: [2,4], 83: [3,5], 84: [2,4,6], 85: [1,3,5], 86: [0], 87: [1], 88: [2,-1], 89: [-1], 90: [-1], 91: [-1], 92: [-1], 93: [-1], 94: [-1], 95: [-1], 96: [-1], 97: [-1], 98: [-1], 99: [-1], 100: [-1], 101: [-1], 102: [-1], 103: [-1]};

const ATOMIC_MASS = {1: 1.008, 2: 4.003, 3: 6.941, 4: 9.012, 5: 10.812, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 10: 20.18, 11: 22.99, 12: 24.305, 13: 26.982, 14: 28.086, 15: 30.974, 16: 32.067, 17: 35.453, 18: 39.948, 19: 39.098, 20: 40.078, 21: 44.956, 22: 47.867, 23: 50.944, 24: 51.996, 25: 54.938, 26: 55.845, 27: 58.933, 28: 58.693, 29: 63.546, 30: 65.39, 31: 69.723, 32: 72.61, 33: 74.922, 34: 78.96, 35: 79.904, 36: 83.8, 37: 85.468, 38: 87.62, 39: 88.906, 40: 91.224, 41: 92.906, 42: 95.94, 43: 98.0, 44: 101.07, 45: 102.906, 46: 106.42, 47: 107.868, 48: 112.412, 49: 114.818, 50: 118.711, 51: 121.76, 52: 127.6, 53: 126.904, 54: 131.29, 55: 132.905, 56: 137.328, 57: 138.906, 58: 140.116, 59: 140.908, 60: 144.24, 61: 145.0, 62: 150.36, 63: 151.964, 64: 157.25, 65: 158.925, 66: 162.5, 67: 164.93, 68: 167.26, 69: 168.934, 70: 173.04, 71: 174.967, 72: 178.49, 73: 180.948, 74: 183.84, 75: 186.207, 76: 190.23, 77: 192.217, 78: 195.078, 79: 196.967, 80: 200.59, 81: 204.383, 82: 207.2, 83: 208.98, 84: 209.0, 85: 210.0, 86: 222.0, 87: 223.0, 88: 226.0, 89: 227.0, 90: 232.038, 91: 231.036, 92: 238.029, 93: 237.0, 94: 244.0, 95: 243.0, 96: 247.0, 97: 247.0, 98: 251.0, 99: 252.0, 100: 257.0, 101: 258.0, 102: 259.0, 103: 262.0};

const ELEMENT_SYMBOL = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr'};

/**
 * Exact masses for isotopes, keyed "atomicNumber_massNumber".
 *
 * An explicitly isotope-labelled atom takes its real mass, not its mass number - RDKit
 * reports 2.014102 for deuterium, not 2. Anything not listed falls back to the mass
 * number, the right approximation for an isotope nobody has tabulated.
 */
const ISOTOPE_MASS = {'1_2': 2.014101778, '1_3': 3.016049278, '5_10': 10.012937, '5_11': 11.0093054, '6_13': 13.00335484, '6_14': 14.00324199, '7_15': 15.0001089, '8_17': 16.9991317, '8_18': 17.999161, '9_18': 18.000938, '15_32': 31.97390727, '16_33': 32.97145876, '16_34': 33.9678669, '16_35': 34.96903216, '16_36': 35.96708076, '17_36': 35.96830698, '17_37': 36.96590259, '35_79': 78.9183371, '35_81': 80.9162906, '53_123': 122.905589, '53_124': 123.9062099, '53_125': 124.9046302, '53_127': 126.904473, '53_131': 130.9061246, '34_77': 76.919914, '14_29': 28.9764947, '14_30': 29.97377017};

// == Ports of RDKit algorithms ==
//
// RDKit.js has no per-atom API at all - `get_json()` gives atomic numbers, charges,
// hydrogen counts, radicals, kekulized bond orders and an aromaticity/ring extension
// block, and `get_descriptors()` returns molecule-level scalars. Several of the atom and
// bond properties the MEGAN featurizer needs are therefore not retrievable and have to be
// recomputed here.
//
// These are transcriptions of the RDKit C++ sources, not approximations:
//
//   setConjugation, isAtomConjugCand, markConjAtomBonds  Code/GraphMol/ConjugHybrid.cpp
//   countAtomElec                                        Code/GraphMol/Aromaticity.cpp
//   calculateExplicitValence, getEffectiveAtomicNum      Code/GraphMol/Atom.cpp
//
// Verified against RDKit over ~429,000 bonds - all of AqSolDB, lipophilicity and logp,
// plus hand-picked adversarial chemistry - with zero disagreements. Earlier attempts to
// infer the conjugation rule from its behaviour reached only 79-96%, because the real
// algorithm is pairwise and depends on RDKit's aromatic valence clamp; neither is
// guessable from outputs.

/** Bond-order contribution to valence. An aromatic bond counts 1.5, as in RDKit. */
function valenceContrib(bond) {
  return bond.aromatic ? 1.5 : bond.order;
}

/**
 * Port of RDKit's calculateExplicitValence.
 *
 * The subtlety is the aromatic clamp: an aromatic carbon carrying an exocyclic double
 * bond accumulates 1.5 + 1.5 + 2.0 = 5.0, but RDKit reports 4, because an aromatic atom
 * whose accumulated valence exceeds the default is pulled down to the nearest allowed
 * valence within 1.5. Skipping it makes every such atom look hypervalent, which silently
 * drops it out of conjugation.
 *
 * @param {object} atom {z, chg, aromatic}
 * @param {number} accum summed bond contributions, plus hydrogens if they should count
 */
function calcExplicitValence(atom, accum) {
  // A charged atom is treated as the isoelectronic neutral element.
  const ownValences = VALENCE_LIST[atom.z] ?? [-1];
  let effectiveZ = atom.z;
  if (ownValences.length > 1 || ownValences[0] !== -1) {
    effectiveZ = Math.max(0, Math.min(atom.z - (atom.chg || 0), 103));
  }

  const dv = DEFAULT_VALENCE[effectiveZ] ?? -1;
  const valences = VALENCE_LIST[effectiveZ] ?? [-1];

  if (dv >= 0 && accum > dv && atom.aromatic) {
    let pval = dv;
    for (const val of valences) {
      if (val === -1 || val > accum) break;
      pval = val;
    }
    if (accum - pval <= 1.5) accum = pval;
  }

  // RDKit biases by 0.1 so that an x.5 sum rounds up rather than to even.
  return Math.round(accum + 0.1);
}

/**
 * Port of MolOps::countAtomElec - electrons the atom can donate to a pi system.
 * Returns -1 for univalent atoms and anything more than three-coordinate, which is how
 * RDKit excludes them from conjugation outright.
 */
function countAtomElec(atom, degree, explicitValence) {
  const dv = DEFAULT_VALENCE[atom.z] ?? -1;
  if (dv <= 1) return -1;

  const totalDegree = degree + atom.impHs;
  if (totalDegree > 3) return -1;

  let nLonePairs = (N_OUTER_ELECS[atom.z] ?? 0) - dv;
  nLonePairs = Math.max(nLonePairs - (atom.chg || 0), 0);

  let res = (dv - totalDegree) + nLonePairs - (atom.nRad || 0);

  if (res > 1) {
    // A triple bond or higher donates only one electron.
    if (explicitValence - degree > 1) res = 1;
  }
  return res;
}

/** Port of isAtomConjugCand. */
function isConjugCandidate(atom, degree, explicitValence, totalValence) {
  const valences = VALENCE_LIST[atom.z] ?? [-1];

  // Hypervalent neutral atoms do not conjugate.
  if (!atom.chg && valences.length && valences[0] >= 0 && totalValence > valences[0]) {
    return false;
  }

  const nOuter = N_OUTER_ELECS[atom.z] ?? 0;
  const totalDegree = degree + atom.impHs;

  // RDKit's Issue211 fix: keeps group 15/16 atoms beyond the first row (the P in
  // Pc1ccccc1) from picking up conjugation and turning SP2.
  const ok = atom.z <= 10
    || (nOuter !== 5 && nOuter !== 6)
    || (nOuter === 6 && totalDegree < 2);

  return ok && countAtomElec(atom, degree, explicitValence) > 0;
}

/**
 * Build per-atom topology: heavy-atom degree, summed bond contributions, incident bonds.
 */
function topology(atoms, bonds) {
  const degree = new Array(atoms.length).fill(0);
  const accum = new Array(atoms.length).fill(0);
  const incident = atoms.map(() => []);

  bonds.forEach((bond, index) => {
    const contrib = valenceContrib(bond);
    for (const [a, other] of [[bond.i, bond.j], [bond.j, bond.i]]) {
      degree[a] += 1;
      accum[a] += contrib;
      incident[a].push({ other, bond, index });
    }
  });

  return { degree, accum, incident };
}

/**
 * Both valences RDKit uses, for one atom.
 *
 * RDKit adds only the EXPLICIT hydrogens before the aromatic clamp and the implicit ones
 * after. CommonChem reports a single hydrogen count and hides that split - an aromatic
 * N-H stores its hydrogen as explicit while an aromatic C-H stores it as implicit, and
 * get_json() reports "impHs: 1" for both. Feeding all hydrogens in before the clamp
 * reaches the same total in every case that can physically occur, because the clamp only
 * fires once the sum already exceeds the default valence.
 */
function atomValences(atom, accum) {
  return {
    explicit: calcExplicitValence(atom, accum),
    total: calcExplicitValence(atom, accum + atom.impHs),
  };
}

/**
 * Port of MolOps::setConjugation.
 *
 * Note this is a PAIRWISE rule: a multiple bond and a second bond at the same atom are
 * marked together. That is why ethylene's lone C=C is not conjugated - with implicit
 * hydrogens there is no second bond at either carbon to pair it with.
 *
 * @returns {boolean[]} one flag per bond, in input order
 */
function setConjugation(atoms, bonds) {
  const { degree, accum, incident } = topology(atoms, bonds);
  const valences = atoms.map((atom, i) => atomValences(atom, accum[i]));

  // Aromatic bonds start conjugated; everything else starts false.
  const conjugated = bonds.map((bond) => Boolean(bond.aromatic));

  const candidate = atoms.map((atom, i) =>
    isConjugCandidate(atom, degree[i], valences[i].explicit, valences[i].total));
  const numSubstituents = atoms.map((atom, i) =>
    (candidate[i] ? degree[i] + atom.impHs : 0));

  for (let idx = 0; idx < atoms.length; idx++) {
    if (!candidate[idx]) continue;
    // markConjAtomBonds requires two or three substituents.
    if (numSubstituents[idx] < 2 || numSubstituents[idx] > 3) continue;

    for (const first of incident[idx]) {
      // The first bond must contribute a pi system, and its far end must take part.
      if (valenceContrib(first.bond) < 1.5 || !candidate[first.other]) continue;

      for (const second of incident[idx]) {
        if (second.index === first.index) continue;
        if (numSubstituents[second.other] > 3) continue;
        if (candidate[second.other]) {
          conjugated[first.index] = true;
          conjugated[second.index] = true;
        }
      }
    }
  }

  return conjugated;
}

/**
 * Port of numBondsPlusLonePairs from ConjugHybrid.cpp.
 */
function numBondsPlusLonePairs(atom, degree, totalValence) {
  const deg = degree + atom.impHs;
  if (atom.z <= 1) return deg;

  const nOuter = N_OUTER_ELECS[atom.z] ?? 0;
  const chg = atom.chg || 0;
  const free = nOuter - (totalValence + chg);

  if (totalValence + nOuter - chg < 8) {
    // Below an octet, radicals have to be accounted for separately.
    const radicals = atom.nRad || 0;
    return deg + Math.trunc((free - radicals) / 2) + radicals;
  }
  return deg + Math.trunc(free / 2);
}

/**
 * Port of MolOps::setHybridization from ConjugHybrid.cpp.
 *
 * The SP3-to-SP2 demotion at four orbitals is driven by whether the atom carries a
 * CONJUGATED bond, so hybridization depends on setConjugation having run first. That is
 * what makes an amide nitrogen SP2 - and it is exact, where a "lone pair next to a pi
 * system" heuristic only ever approximated it (99.93% of atoms against 100% here).
 *
 * Verified against RDKit over 409,678 atoms - AqSolDB, lipophilicity and logp - with zero
 * disagreements.
 *
 * @returns {string[]} 'S' | 'SP' | 'SP2' | 'SP3' | 'SP3D' | 'SP3D2' | 'UNSPECIFIED'
 */
function hybridizations(atoms, bonds) {
  const { degree, accum, incident } = topology(atoms, bonds);
  const totalValence = atoms.map((atom, i) => atomValences(atom, accum[i]).total);

  const conjugated = setConjugation(atoms, bonds);
  const hasConjugatedBond = atoms.map(() => false);
  bonds.forEach((bond, k) => {
    if (conjugated[k]) {
      hasConjugatedBond[bond.i] = true;
      hasConjugatedBond[bond.j] = true;
    }
  });

  return atoms.map((atom, idx) => {
    if (atom.z === 0) return 'UNSPECIFIED';

    const totalDegree = degree[idx] + atom.impHs;

    // A tetrahedral stereocentre with four substituents is SP3 outright.
    if ((atom.chiral === 'cw' || atom.chiral === 'ccw') && totalDegree === 4) {
      return 'SP3';
    }

    const norbs = atom.z < 89
      ? numBondsPlusLonePairs(atom, degree[idx], totalValence[idx])
      : totalDegree;

    switch (norbs) {
      case 0:
      case 1: return 'S';
      case 2: return 'SP';
      case 3: return 'SP2';
      case 4:
        // Issue276: an atom with more than three substituents stays SP3 even when
        // conjugated, so the phosphorus in CP1(C)=CC=CN=C1C is not demoted.
        return (totalDegree > 3 || !hasConjugatedBond[idx]) ? 'SP3' : 'SP2';
      case 5: return 'SP3D';
      case 6: return 'SP3D2';
      default: return 'UNSPECIFIED';
    }
  });
}

// == Featurizer ==

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

/** The `specVersion` of featurization.json that this file reads. */
const SPEC_VERSION = 1;

export class UnsupportedSpecError extends Error {}

/**
 * The remedy for a contract this file cannot read: the export folder's own copy of
 * megan.js, which the package version that wrote the contract made alongside it.
 */
function writtenBy(contract) {
  return `It was written by ${contract?.generatedBy ?? 'another version'}; use the ` +
    'megan.js from the same export folder.';
}

export class Featurizer {
  /**
   * @param {object} RDKit an initialised RDKit.js module
   * @param {object} spec the parsed `featurization.json`
   */
  constructor(RDKit, spec) {
    this.RDKit = RDKit;
    this.spec = spec;

    // Checked first, so that a contract from a newer version is reported as such rather
    // than as whatever part of it this version happens to trip over.
    if (spec?.specVersion !== SPEC_VERSION) {
      throw new UnsupportedSpecError(
        `featurization.json has specVersion ${spec?.specVersion}, but this megan.js ` +
        `reads version ${SPEC_VERSION}. ${writtenBy(spec)}`);
    }

    for (const [encoders, supported, kind] of [
      [spec.nodeAttributes.encoders, SUPPORTED_ATOM_PROPERTIES, 'atom'],
      [spec.edgeAttributes.encoders, SUPPORTED_BOND_PROPERTIES, 'bond'],
    ]) {
      for (const encoder of encoders) {
        if (!supported.has(encoder.property)) {
          throw new UnsupportedSpecError(
            `featurization.json asks for the ${kind} property "${encoder.property}", ` +
            `which this megan.js does not implement. Supported: ` +
            `${[...supported].join(', ')}. ${writtenBy(spec)}`);
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

// == Running a model ==

/** The `schemaVersion` of manifest.json that this file reads. */
const MANIFEST_SCHEMA_VERSION = 1;

/**
 * An exported model, ready to predict molecules from their SMILES.
 *
 * Load one with `MeganModel.load` from an export folder served over HTTP, or build one
 * with the constructor from a manifest and a session created by hand - the way to go in
 * Node, whose `fetch` cannot read local files.
 */
export class MeganModel {
  /**
   * Fetches an export folder's `manifest.json` and `model.onnx` and starts a session.
   *
   * @param {string|URL} folder the export folder, such as 'onnx_export/'. A relative URL
   *   resolves against the page; outside a browser it has to be absolute.
   * @param {object} runtimes
   * @param {object} runtimes.RDKit an initialised RDKit.js module
   * @param {object} runtimes.ort the onnxruntime-web (or onnxruntime-node) module
   * @param {object} [runtimes.sessionOptions] options for `ort.InferenceSession.create`
   * @returns {Promise<MeganModel>}
   */
  static async load(folder, { RDKit, ort, sessionOptions } = {}) {
    requireRuntimes(RDKit, ort);

    // Without the trailing slash, 'onnx_export' and 'model.onnx' would resolve to a
    // model.onnx next to the folder instead of inside it.
    let root;
    try {
      root = new URL(String(folder).replace(/\/?$/, '/'), globalThis.location?.href);
    } catch {
      throw new TypeError(
        `"${folder}" is not an absolute URL. Outside a browser, pass one, or read the ` +
        'files yourself and use the MeganModel constructor.');
    }
    const get = async (name) => {
      const response = await fetch(new URL(name, root));
      if (!response.ok) {
        throw new Error(`Could not load ${response.url}: HTTP ${response.status}`);
      }
      return response;
    };

    const manifest = await (await get('manifest.json')).json();
    checkManifest(manifest);
    const bytes = new Uint8Array(await (await get(manifest.model.path)).arrayBuffer());
    const session = await ort.InferenceSession.create(bytes, sessionOptions);
    return new MeganModel({ manifest, session, RDKit, ort });
  }

  /**
   * @param {object} parts
   * @param {object} parts.manifest the parsed `manifest.json`
   * @param {object} parts.session an `ort.InferenceSession` of the export's `model.onnx`
   * @param {object} parts.RDKit an initialised RDKit.js module
   * @param {object} parts.ort the onnxruntime module the session came from
   */
  constructor({ manifest, session, RDKit, ort }) {
    requireRuntimes(RDKit, ort);
    checkManifest(manifest);

    this.manifest = manifest;
    this.session = session;
    this.ort = ort;
    // The manifest embeds the same contract as featurization.json, so a loaded model
    // needs only the two files.
    this.featurizer = new Featurizer(RDKit, manifest.featurization);
    this.numChannels = manifest.architecture.config.num_channels;
  }

  /**
   * Predicts a batch of molecules with one run of the session.
   *
   * @param {string[]} smilesList
   * @returns {Promise<object[]>} one result per molecule, in input order, with
   *   - `smiles`: the input.
   *   - `prediction`: one value per target. A classifier's logits are turned into
   *     probabilities - by a softmax, or per target by a sigmoid for prediction_mode
   *     'bce' - and regression values are returned as they are.
   *   - `nodeImportance`: one row per atom, in RDKit's atom order for the same SMILES,
   *     with one value per explanation channel. This is `node_importance_norm`, the
   *     output an explanation drawing should use.
   *   - `edgeImportance`: one row per edge, from `edge_importance_norm`. Every bond is
   *     two edges, forward then reverse, as listed in `graph.edgeIndices`.
   *   - `outputs`: every output of the model for this molecule, by its ONNX name.
   *   - `graph`: the featurized molecule, as `Featurizer.process` returns it.
   */
  async predict(smilesList) {
    if (!Array.isArray(smilesList)) {
      throw new TypeError('predict takes an array of SMILES, for example predict(["CCO"]).');
    }
    if (smilesList.length === 0) return [];

    const graphs = smilesList.map((smiles) => this.featurizer.process(smiles));
    const outputs = await this.session.run(
      Featurizer.toFeeds(graphs, this.ort, this.numChannels));

    // The outputs are concatenated over the batch: graph* have one row per molecule,
    // node* one row per atom and edge* one row per edge.
    let node = 0;
    let edge = 0;
    return graphs.map((graph, index) => {
      const own = {};
      for (const [name, tensor] of Object.entries(outputs)) {
        if (name.startsWith('node')) {
          own[name] = rows(tensor, node, graph.nodeAttributes.length);
        } else if (name.startsWith('edge')) {
          own[name] = rows(tensor, edge, graph.edgeIndices.length);
        } else {
          own[name] = rows(tensor, index, 1)[0];
        }
      }
      node += graph.nodeAttributes.length;
      edge += graph.edgeIndices.length;

      return {
        smiles: smilesList[index],
        prediction: activate(own.graph_output, this.manifest.architecture.config.prediction_mode),
        nodeImportance: own.node_importance_norm,
        edgeImportance: own.edge_importance_norm,
        outputs: own,
        graph,
      };
    });
  }
}

/** Refuses a manifest that no model can be run from with this file, saying why. */
function checkManifest(manifest) {
  if (manifest?.schemaVersion !== MANIFEST_SCHEMA_VERSION) {
    throw new UnsupportedSpecError(
      `manifest.json has schemaVersion ${manifest?.schemaVersion}, but this megan.js ` +
      `reads version ${MANIFEST_SCHEMA_VERSION}. ${writtenBy(manifest)}`);
  }
  if (!manifest.featurization) {
    throw new UnsupportedSpecError(
      'This export has no featurization contract, so its inputs cannot be computed from ' +
      `SMILES: ${manifest.featurizationOmittedBecause ?? 'no reason was recorded'}. ` +
      'Retrain the model on the portable feature set to run it from JavaScript.');
  }
}

function requireRuntimes(RDKit, ort) {
  if (typeof RDKit?.get_mol !== 'function') {
    throw new TypeError('Pass an initialised RDKit.js module as RDKit.');
  }
  if (typeof ort?.Tensor !== 'function') {
    throw new TypeError('Pass the onnxruntime-web or onnxruntime-node module as ort.');
  }
}

/**
 * Rows `start` to `start + count` of an output tensor, each as nested arrays over the
 * remaining dimensions - a slice of a (num_nodes, K) output becomes one K-array per atom.
 */
function rows(tensor, start, count) {
  const inner = tensor.dims.slice(1);
  const size = inner.reduce((a, b) => a * b, 1);
  return Array.from({ length: count }, (_, k) => nest(tensor.data, inner, (start + k) * size));
}

function nest(data, dims, offset) {
  if (dims.length === 0) return Number(data[offset]);
  const stride = dims.slice(1).reduce((a, b) => a * b, 1);
  return Array.from({ length: dims[0] }, (_, i) => nest(data, dims.slice(1), offset + i * stride));
}

/**
 * A classifier's logits as probabilities. MEGAN applies no output activation, so
 * `graph_output` holds logits for classification and raw values for regression.
 */
function activate(values, predictionMode) {
  if (predictionMode === 'classification') {
    // Shifted by the maximum, so that a large logit cannot overflow Math.exp.
    const max = Math.max(...values);
    const exps = values.map((value) => Math.exp(value - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((value) => value / sum);
  }
  if (predictionMode === 'bce') {
    return values.map((value) => 1 / (1 + Math.exp(-value)));
  }
  return values;
}
