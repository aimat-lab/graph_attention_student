# ADR-004: Ship the JavaScript Runtime in Every Export, Not as an npm Package

**Date:** 2026-10-20
**Status:** Accepted

## Context

ADR-002 ships a reference JavaScript featurizer with the ONNX export. It was three ES
modules in `js/` (featurizer, RDKit algorithm ports, periodic table), and a consumer had to
copy all three next to their page, together with the RDKit.js and ONNX Runtime Web
runtime files, and then write the same code every time: read the channel count from the
manifest, batch the inputs, split the flat outputs back into molecules, and turn a
classifier's logits into probabilities.

The obvious way to distribute JavaScript is an npm package. But the hard problem here is
not distribution: the featurizer has to match the featurization contract of the package
version that exported the model, and an npm package adds a second version line that has
to be kept in step with the Python one. It would also need an npm account or organisation
that someone in the lab owns, a publish job, and the stability promise npm users expect -
for an API that changed incompatibly while this was being written (`toFeeds` stopped
guessing the channel count). Nothing in the code needs a package manager: it has no
dependencies of its own, because RDKit.js and ONNX Runtime Web are passed in.

## Decision

- The three modules become one file, `megan.js`, with a `MeganModel` class on top:
  `MeganModel.load(folder, { RDKit, ort })` and `model.predict(smilesList)`, which returns
  each molecule's prediction (probabilities for a classifier) and per-atom importances.
  The featurizer and `toFeeds` stay exported for callers that run the session themselves.
- The file lives inside the Python package, next to `portable_processing.py`, which is its
  Python counterpart. Wheels, sdists and editable installs all have it at the same path,
  `MEGAN_JS_PATH`, with no packaging configuration.
- Every export that writes a `featurization.json` copies `megan.js` beside it. The copy is
  from the package version that wrote the contract, so the two cannot disagree, and the
  folder can be served as a static site as it is.
- For pages that load it from elsewhere, jsDelivr serves the file straight from each
  release tag (`cdn.jsdelivr.net/gh/aimat-lab/graph_attention_student@v<version>/...`),
  with a JavaScript content type and permanent caching. Releases already push the tag, so
  this costs nothing extra.
- `megan.js` checks `specVersion` of the featurization contract and `schemaVersion` of the
  manifest, and refuses one it cannot read with an error naming the version that wrote it.
  A copy from the wrong version therefore fails loudly rather than featurizing wrongly.

## Consequences

**Good**

- A browser page needs nothing next to it but the export folder; RDKit.js and ONNX Runtime
  Web load from their own npm packages through a CDN.
- There is no second release process and no version mapping between the Python package
  and a JavaScript one.
- Publishing to npm later stays cheap: the file is already an ES module without
  dependencies, and jsDelivr and unpkg serve npm packages too, so CDN users would only
  change the URL.

**Bad / accepted costs**

- No `npm install`, no TypeScript declarations and no semver updates for applications built
  with a bundler; they vendor the file instead.
- The jsDelivr URL contains the package path, and it only works once the release tag
  exists. The first request after a new tag is slow (over 20 s when measured) while
  jsDelivr fetches the repository.
- A bug fix in `megan.js` does not reach existing export folders; they keep the copy they
  were written with.
- Outside a `"type": "module"` package, Node loads the file as an ES module only by syntax
  detection, which Node 22 does by default; CI therefore runs Node 22.

## Alternatives considered

**An npm package.** The conventional choice, and the right one once a consumer builds an
application with a bundler and wants types and updates. Rejected for now for the reasons
above, without closing the door.

**A CDN link only, without the copy in each export.** Leaves every consumer to pick the
matching version by hand, which is exactly the mismatch the copy rules out.

**Keeping the file in `js/` and adding it to the wheel with a forced include.** Gives a
shorter CDN URL, but an editable install imports the package from the source tree, where
the file would not sit next to the exporter, so it would need a second lookup path that
only release builds exercise.
