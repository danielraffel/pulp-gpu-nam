#!/usr/bin/env bash
# Create a pulp-gpu-nam worktree with its Pulp submodule ready, without
# downloading Pulp again.
#
# A plain `git submodule update --init --recursive` in a fresh worktree or
# clone fetches the whole Pulp repository from GitHub, which is slow and fails
# on any network blip. This script keeps GitHub as the submodule's origin but
# borrows every object that already exists on this machine (--reference), so
# only what is genuinely missing is fetched. --dissociate then copies the
# borrowed objects in, so the new worktree never depends on another checkout's
# object store staying intact.
#
# Worktrees are placed under $PULP_WORKTREES_ROOT when set, otherwise next to
# this checkout. /tmp is refused: macOS cleans it, and a reboot loses the work.
#
# Usage:  scripts/new-worktree.sh <name> [<ref>] [-b <new-branch>]
#   <name>    directory name (or absolute path) for the worktree
#   <ref>     commit or branch to check out (default: origin/main)
#   -b NAME   create NAME as a new branch at <ref>
# Env:
#   PULP_REFERENCE_REPO  a Pulp repository (worktree or .git dir) to borrow
#                        objects from; auto-detected when unset
set -euo pipefail

usage() { sed -n '/^# Usage:/,/^# Env:/p' "$0" | sed '$d; s/^# \{0,1\}//' >&2; exit 2; }

name="" ref="origin/main" branch=""
while (($#)); do
  case "$1" in
    -b) [[ $# -ge 2 ]] || usage; branch="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) if [[ -z "$name" ]]; then name="$1"; else ref="$1"; fi; shift ;;
  esac
done
[[ -n "$name" ]] || usage

here="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
common="$(cd "$here" && cd "$(git rev-parse --git-common-dir)" && pwd)"

case "$name" in
  /*) dest="$name" ;;
  *)  dest="${PULP_WORKTREES_ROOT:-$(dirname "$here")}/$name" ;;
esac
case "$(cd "$(dirname "$dest")" 2>/dev/null && pwd -P)/" in
  /tmp/*|/private/tmp/*|/var/folders/*|/private/var/folders/*)
    echo "error: refusing to create a worktree under a temporary directory ($dest)." >&2
    echo "       macOS cleans it and a reboot loses the work; set PULP_WORKTREES_ROOT or pass a path elsewhere." >&2
    exit 1 ;;
esac
[[ ! -e "$dest" ]] || { echo "error: $dest already exists" >&2; exit 1; }

# Borrow objects from, in order: an explicit repository, this checkout's own
# initialised submodule store, then a sibling Pulp checkout.
reference=""
for candidate in "${PULP_REFERENCE_REPO:-}" "$common/modules/pulp" "$(dirname "$here")/pulp"; do
  [[ -n "$candidate" ]] || continue
  if git -C "$candidate" rev-parse --git-dir >/dev/null 2>&1; then
    reference="$(cd "$candidate" && cd "$(git rev-parse --git-dir)" && pwd)"
    break
  fi
done

git -C "$here" fetch --quiet origin || echo "warning: could not fetch origin; using local refs" >&2
if [[ -n "$branch" ]]; then
  git -C "$here" worktree add -b "$branch" "$dest" "$ref"
else
  git -C "$here" worktree add --detach "$dest" "$ref"
fi

# Use each submodule's canonical URL from .gitmodules for this command only. A
# checkout may override it in its shared config (for example to a local partial
# clone), which new worktrees inherit and which can lack the objects they need.
url_overrides=()
while read -r key url; do
  sub="${key#submodule.}"; sub="${sub%.url}"
  url_overrides+=(-c "submodule.${sub}.url=${url}")
done < <(git -C "$dest" config -f .gitmodules --get-regexp '^submodule\..*\.url$' || true)

if [[ -n "$reference" ]]; then
  echo "borrowing Pulp objects from $reference"
  git -C "$dest" "${url_overrides[@]}" submodule update --init --recursive \
    --reference "$reference" --dissociate
else
  echo "warning: no local Pulp repository found to borrow from; fetching Pulp from its origin" >&2
  git -C "$dest" "${url_overrides[@]}" submodule update --init --recursive
fi
echo "ready: $dest"
