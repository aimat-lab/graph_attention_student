#!/usr/bin/env bash
# release.sh - Automated release script for graph_attention_student
#
# Usage: ./release.sh
#
# Bump the version manually beforehand (edit graph_attention_student/VERSION,
# pyproject.toml and README.rst); this script reads the version from
# graph_attention_student/VERSION.
#
# This script:
#   1. Validates preconditions (clean tree, on master, tools available)
#   2. Runs nox tests
#   3. Commits any pending version files and tags locally
#   4. Builds and publishes the package via uv
#
# It does NOT push to the remote or create a GitHub release — do those manually.

set -euo pipefail

# ── Colors & helpers ─────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Precondition checks ─────────────────────────────────────────────

step "Checking preconditions"

# Check required tools
for cmd in git nox uv; do
    if ! command -v "$cmd" &>/dev/null; then
        error "Required tool '$cmd' is not installed or not in PATH."
    fi
done
ok "All required tools are available"

# Must be on master branch
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "master" ]]; then
    error "Must be on 'master' branch to release (currently on '$CURRENT_BRANCH')."
fi
ok "On branch 'master'"

# Working tree should be clean
if [[ -n "$(git status --porcelain)" ]]; then
    warn "Working tree is dirty. Uncommitted changes will NOT be included in the release."
    git status --short
    echo ""
    read -rp "Continue anyway? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        error "Aborted by user."
    fi
else
    ok "Working tree is clean"
fi

# Read the (manually-set) version to release
NEW_VERSION=$(cat graph_attention_student/VERSION)
info "Releasing version: ${BOLD}$NEW_VERSION${NC}"

# ── Run tests ────────────────────────────────────────────────────────

step "Running nox tests"
info "Executing: nox -s test"

if ! nox -s test; then
    error "Tests failed. Aborting release."
fi
ok "All tests passed"

# ── Git commit & tag ─────────────────────────────────────────────────

TAG_NAME="v${NEW_VERSION}"

step "Creating git commit and tag"

info "Staging version files..."
git add pyproject.toml graph_attention_student/VERSION README.rst

if git diff --cached --quiet; then
    info "No pending version changes to commit (already committed manually)."
else
    info "Committing version files..."
    git commit -m "$(cat <<EOF
${NEW_VERSION}

Release ${NEW_VERSION}

EOF
)"
    ok "Committed"
fi

info "Creating tag '${TAG_NAME}'..."
git tag -a "$TAG_NAME" -m "Release ${NEW_VERSION}"
ok "Tag '${TAG_NAME}' created"

# ── Build & publish ──────────────────────────────────────────────────

step "Building package"

info "Cleaning previous build artifacts..."
rm -rf dist/

info "Executing: uv build"
uv build
ok "Package built"

step "Publishing package"

info "Executing: uv publish"
uv publish dist/*"${NEW_VERSION}"*
ok "Package published"

# ── Done ─────────────────────────────────────────────────────────────

step "Release ${NEW_VERSION} complete!"
echo ""
info "Summary:"
info "  Version:  ${NEW_VERSION}"
info "  Tag:      ${TAG_NAME} (local — push manually with: git push origin master && git push origin ${TAG_NAME})"
echo ""
