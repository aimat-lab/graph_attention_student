#!/usr/bin/env bash
# release.sh - Automated release script for graph_attention_student
#
# Usage: ./release.sh <major|minor|patch>
#
# This script:
#   1. Validates preconditions (clean tree, on master, tools available)
#   2. Runs nox tests
#   3. Bumps the version via bump-my-version
#   4. Commits, tags, and pushes
#   5. Creates a GitHub release
#   6. Builds and publishes the package via uv

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

# ── Argument validation ──────────────────────────────────────────────

BUMP_PART="${1:-}"

if [[ -z "$BUMP_PART" ]]; then
    echo "Usage: ./release.sh <major|minor|patch>"
    exit 1
fi

if [[ "$BUMP_PART" != "major" && "$BUMP_PART" != "minor" && "$BUMP_PART" != "patch" ]]; then
    error "Invalid bump part '$BUMP_PART'. Must be one of: major, minor, patch"
fi

# ── Precondition checks ─────────────────────────────────────────────

step "Checking preconditions"

# Check required tools
for cmd in git nox bump-my-version gh uv; do
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

# Read current version before bump
OLD_VERSION=$(cat graph_attention_student/VERSION)
info "Current version: ${BOLD}$OLD_VERSION${NC}"

# ── Run tests ────────────────────────────────────────────────────────

step "Running nox tests"
info "Executing: nox -s test"

if ! nox -s test; then
    error "Tests failed. Aborting release."
fi
ok "All tests passed"

# ── Bump version ─────────────────────────────────────────────────────

step "Bumping version ($BUMP_PART)"
info "Executing: bump-my-version bump $BUMP_PART"

bump-my-version bump "$BUMP_PART"

NEW_VERSION=$(cat graph_attention_student/VERSION)
ok "Version bumped: ${OLD_VERSION} → ${BOLD}${NEW_VERSION}${NC}"

# ── Git commit & tag ─────────────────────────────────────────────────

TAG_NAME="v${NEW_VERSION}"

step "Creating git commit and tag"

info "Staging changed files..."
git add pyproject.toml graph_attention_student/VERSION README.rst

info "Committing version bump..."
git commit -m "$(cat <<EOF
${NEW_VERSION}

Bump version: ${OLD_VERSION} → ${NEW_VERSION}

EOF
)"
ok "Committed"

info "Creating tag '${TAG_NAME}'..."
git tag -a "$TAG_NAME" -m "Release ${NEW_VERSION}"
ok "Tag '${TAG_NAME}' created"

# ── Git push ─────────────────────────────────────────────────────────

step "Pushing to remote"

info "Pushing commits..."
git push origin master

info "Pushing tags..."
git push origin "$TAG_NAME"
ok "Pushed commits and tag to origin"

# ── GitHub release ───────────────────────────────────────────────────

step "Creating GitHub release"

REPO_URL=$(git remote get-url origin | sed 's/\.git$//' | sed 's|git@github.com:|https://github.com/|')
# Extract owner/repo from origin URL for gh commands (avoids gh picking up the wrong remote)
GH_REPO=$(echo "$REPO_URL" | sed 's|https://github.com/||')
CHANGELOG_URL="${REPO_URL}/blob/${TAG_NAME}/CHANGELOG.md"

info "Creating release '${TAG_NAME}' on GitHub (repo: ${GH_REPO})..."
gh release create "$TAG_NAME" \
    --repo "$GH_REPO" \
    --title "${NEW_VERSION}" \
    --notes "$(cat <<EOF
## ${NEW_VERSION}

See the full changelog: [CHANGELOG.md](${CHANGELOG_URL})
EOF
)"
ok "GitHub release created"

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
info "  Version:  ${OLD_VERSION} → ${NEW_VERSION}"
info "  Tag:      ${TAG_NAME}"
info "  Release:  ${REPO_URL}/releases/tag/${TAG_NAME}"
echo ""
