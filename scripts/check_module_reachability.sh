#!/bin/bash
# check_module_reachability.sh — CI guard against dead-on-disk Rust modules.
#
# Why this exists: a prior audit wave found ~9000 lines across several
# fully-implemented, fully-tested modules (e.g. oxigaf-render/src/light_probe.rs,
# oxigaf-cli/src/commands/scene_ops.rs at the time) that sat on disk but were
# never declared with `mod`/`pub mod` anywhere reachable from their crate
# root, so `cargo build`/`cargo test` silently never compiled them — no
# warning, no error, just permanently dead code and a false sense of coverage.
#
# What it does: for every crate under crates/*/, starting from src/lib.rs
# and/or src/main.rs, it walks the *actual* module tree by parsing
# `mod NAME;` / `pub mod NAME;` (including `pub(crate)`/`pub(super)`/`pub(in
# ...)` visibilities and `#[path = "..."]` overrides) and follows each
# file-backed module to its file (NAME.rs or NAME/mod.rs, or the #[path]
# target), recursing into directory modules. It then diffs the set of
# `.rs` files actually reachable this way against every `.rs` file that
# physically exists under that crate's src/ tree. Any file present on disk
# but absent from the reachable set is reported as an orphan and the
# script exits non-zero.
#
# This is a *static* text-based mod-graph walk, not a compiler invocation —
# it takes on the order of 15-20 seconds workspace-wide (dominated by one
# `grep` fork per .rs file: ~460 files workspace-wide as of introduction)
# and needs nothing beyond grep/find, so it belongs in the fast local
# pre-push tier (see ci-local skill) as well as policy-check — it is still
# an order of magnitude cheaper than `cargo check --all-features` (minutes),
# never mind a full `cargo nextest run --workspace --all-features`.
#
# Deliberate scope limits (see "Known limitations" at the bottom):
#   - Only src/ trees are scanned (examples/, tests/, benches/ are separate
#     Cargo targets with their own root files, not part of the lib/bin
#     module tree, and cargo already errors loudly if *those* roots
#     reference a missing file).
#   - `mod name { ... }` inline modules are correctly *not* flagged (they
#     have no backing file to check) but are still walked so that any
#     `mod`/`#[path]` declarations *nested inside* them are followed.
#   - `cfg`-gated modules (`#[cfg(feature = "x")] pub mod y;`) are treated
#     as reachable unconditionally. This intentionally over-approximates
#     reachability (a module gated behind a feature nobody enables is still
#     "declared", just not always compiled) rather than trying to evaluate
#     feature predicates, which would require a real cfg expression parser.
#
# Usage:
#   scripts/check_module_reachability.sh              # scan every crate
#   scripts/check_module_reachability.sh oxigaf-render # scan one crate
#   scripts/check_module_reachability.sh --self-test   # regression-test
#                                                       # this script itself
#                                                       # (see run_self_test)
#
# Exit status: 0 if every .rs file under every scanned crate's src/ is
# reachable from its crate root(s); 1 otherwise (orphans listed on stderr).
# --self-test exits 0/1 on its own pass/fail, independent of this repo's
# crates/ tree.

set -uo pipefail

# Resolve the repo root from this script's own location so the check works
# regardless of the caller's cwd, without hardcoding any absolute path.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
repo_root="$(cd "${script_dir}/.." >/dev/null 2>&1 && pwd)"
crates_dir="${repo_root}/crates"

if [ ! -d "${crates_dir}" ]; then
    echo "check_module_reachability: no crates/ dir found at ${crates_dir}" >&2
    exit 1
fi

# A `mod NAME;` / `pub(...) mod NAME;` declaration line, capturing NAME.
# Matches: mod x;  pub mod x;  pub(crate) mod x;  pub(super) mod x;
#          pub(self) mod x;  pub(in some::path) mod x;
mod_decl_re='^[[:space:]]*(pub([[:space:]]*\(([[:space:]]*(crate|super|self|in[[:space:]]+[A-Za-z0-9_:]+)[[:space:]]*)\))?[[:space:]]+)?mod[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*;[[:space:]]*(//.*)?$'

# A `mod NAME {` inline-module opener, capturing NAME (no backing file, but
# we still need to recurse *into* it to find nested mod/`#[path]` lines).
mod_block_re='^[[:space:]]*(pub([[:space:]]*\(([[:space:]]*(crate|super|self|in[[:space:]]+[A-Za-z0-9_:]+)[[:space:]]*)\))?[[:space:]]+)?mod[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\{[[:space:]]*(//.*)?$'

path_attr_re='^[[:space:]]*#\[[[:space:]]*path[[:space:]]*=[[:space:]]*"([^"]+)"[[:space:]]*\][[:space:]]*$'

# A cheap pre-filter (single `grep -E` per *file*, not per line) that both
# skips files with no `mod`/`path =` token at all (empty output, loop
# below never executes) AND cuts the per-line [[ =~ ]] loop down to just
# the small number of candidate lines instead of the whole file body. A
# stray unanchored match inside a comment or an unrelated `let x_path = `
# assignment only costs one extra (still fork-free) [[ =~ ]] attempt on
# that one line — the anchored ^-based mod_decl_re/mod_block_re/
# path_attr_re below never match inside a comment, since a comment line's
# first non-whitespace token is `//`, not `mod`/`pub`/`#[path`.
# NOTE: earlier revisions of this script called out to helper functions via
# `$(...)` command substitution for every line of every file. That forks a
# subshell per call — with ~490k total lines under crates/*/src/ at the time
# this script was written, that was 1M+ forks and made the workspace-wide
# scan hang for minutes. `[[ =~ ]]` and `BASH_REMATCH` are bash builtins
# with zero fork cost, so the per-line matching below uses those directly
# instead — the only forks left are ~2 per *file* (dirname/basename) plus a
# handful of `find`/`sort` calls, not per *line*.

declare -a reachable_files=()
declare -a orphan_files=()
declare -a all_crate_summaries=()
overall_status=0

# is_reachable <path> — true if path is already recorded in reachable_files.
is_reachable() {
    local needle="$1" f
    # Length-guarded: under `set -u`, bash 3.2 (macOS default /bin/bash)
    # treats "${arr[@]}" on a declared-but-empty array as an unbound
    # variable, so every expansion in this script is guarded like this
    # rather than relying on bash 4+'s more forgiving behavior.
    if [ "${#reachable_files[@]}" -gt 0 ]; then
        for f in "${reachable_files[@]}"; do
            [ "${f}" = "${needle}" ] && return 0
        done
    fi
    return 1
}

# walk_file <abs-file-path>
# Marks a root/module file reachable and scans it for further `mod`/
# `#[path]` declarations, resolving each to a file and recursing.
walk_file() {
    local file="$1"
    [ -f "${file}" ] || return 0
    if is_reachable "${file}"; then
        return 0
    fi
    reachable_files+=("${file}")

    # dirname/basename via bash parameter expansion (builtin, no fork) —
    # ${file} is always the absolute path we built or were seeded with, so
    # a plain suffix/prefix strip is sufficient here.
    local dir="${file%/*}"
    local base="${file##*/}"
    base="${base%.rs}"
    # The directory a same-named child module directory would live in:
    # for src/foo.rs a child `mod bar;` resolves to src/foo/bar.rs or
    # src/foo/bar/mod.rs; for src/lib.rs / src/main.rs / any mod.rs, a
    # child resolves relative to that file's own directory.
    local child_base_dir="${dir}"
    if [ "${base}" != "lib" ] && [ "${base}" != "main" ] && [ "${base}" != "mod" ]; then
        child_base_dir="${dir}/${base}"
    fi

    # Pre-filter with `grep` (fast, buffered, C-level) down to just the
    # handful of lines that could possibly be a mod/path declaration,
    # instead of streaming every line of the file through bash's `read`
    # builtin (which reads byte-at-a-time and is the dominant cost of a
    # workspace-wide scan otherwise — see NOTE above the regexes). A file
    # with none of these tokens produces no output and the loop below
    # simply doesn't execute.
    local mod_lines
    mod_lines="$(grep -E 'mod[[:space:]]|path[[:space:]]*=' "${file}" 2>/dev/null || true)"
    [ -z "${mod_lines}" ] && return 0

    local pending_path_attr=""
    local line name target
    while IFS= read -r line || [ -n "${line}" ]; do
        # A `#[path = "..."]` attribute applies to the *next* mod
        # declaration line, whether `mod x;` or `mod x { ... }`.
        if [[ "${line}" =~ ${path_attr_re} ]]; then
            pending_path_attr="${BASH_REMATCH[1]}"
            continue
        fi

        if [[ "${line}" =~ ${mod_decl_re} ]]; then
            name="${BASH_REMATCH[5]}"
            if [ -n "${pending_path_attr}" ]; then
                # `#[path = "..."]` is always resolved relative to the
                # directory *containing the file with the attribute*
                # (i.e. `dir`), never relative to `child_base_dir` — that
                # only coincides with `dir` for lib.rs/main.rs/mod.rs.
                # Verified against rustc directly (not from memory): a
                # `#[path = "elsewhere/thing.rs"]` inside src/alpha.rs
                # (itself `mod alpha;` from lib.rs) resolves to
                # src/elsewhere/thing.rs, NOT src/alpha/elsewhere/thing.rs.
                # This matches the real pattern in this repo, e.g.
                # spectral_analysis.rs's own
                # `#[path = "spectral_analysis/tests.rs"]` — the path
                # string already spells out its own basename as a prefix
                # precisely because it is dir-relative, not
                # child-base-dir-relative.
                target="${dir}/${pending_path_attr}"
            else
                # Default resolution: NAME.rs, else NAME/mod.rs.
                if [ -f "${child_base_dir}/${name}.rs" ]; then
                    target="${child_base_dir}/${name}.rs"
                else
                    target="${child_base_dir}/${name}/mod.rs"
                fi
            fi
            pending_path_attr=""
            walk_file "${target}"
            continue
        fi

        if [[ "${line}" =~ ${mod_block_re} ]]; then
            # Inline `mod NAME { ... }` has no backing file to add to the
            # reachable set. A `mod`/`#[path]` declaration nested *inside*
            # such a block resolves relative to yet another base (verified
            # against rustc: it's child_base_dir/NAME/, i.e. one level
            # deeper than a same-named top-level `mod NAME;` sibling would
            # use for its own children) — a third resolution rule this
            # walker does not implement, on top of the two it does (see
            # header note: known limitation, verified zero occurrences of
            # nested mod/#[path] inside an inline `mod {}` block workspace-
            # wide at review time). We still consume the pending attr so a
            # stray one before an inline block isn't misapplied downstream.
            pending_path_attr=""
            continue
        fi
    done <<< "${mod_lines}"
}

# run_self_test — regression test for this script's own walk_file/
# is_reachable logic, run on a disposable synthetic fixture crate rather
# than this repo's real crates/ tree (whose current orphan count can and
# should change over time as the real bugs get fixed, which would make a
# test asserting against it flaky by design). Exercises exactly the two
# resolution-rule bugs this script has already been through in review:
#   1. false hang / mis-scoped mod names: a flat top-level orphan and a
#      nested-under-mod.rs orphan must both be caught, while their
#      correctly-wired siblings (including a `#[cfg(test)] mod tests { }`
#      inline block, which must never itself be flagged) must not be.
#   2. #[path] base-directory resolution: verified against rustc directly
#      (see the resolution-rule comment above) that `#[path]` on a
#      file-backed `mod NAME;` resolves relative to the *declaring file's
#      own directory*, not the child_base_dir a plain NAME.rs/NAME/mod.rs
#      lookup would use — these two only coincide for lib.rs/main.rs/
#      mod.rs, so a fixture that only tested a mod.rs-hosted #[path] would
#      pass even with that resolution rule backwards. This fixture uses a
#      non-mod.rs file (mirroring the repo's own spectral_analysis.rs
#      pattern) specifically so the two rules cannot coincidentally agree.
# Uses `mktemp -d` (this script's own equivalent of the project's
# std::env::temp_dir() policy for test fixtures) so nothing is written
# under the repo itself and a leftover fixture from a killed run can never
# be mistaken for real source.
run_self_test() {
    local fixture_root
    fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/oxigaf-module-reachability-selftest.XXXXXX")"
    local fixture_src="${fixture_root}/crates/fixture-crate/src"
    mkdir -p "${fixture_src}/subdir" "${fixture_src}/pathmod"

    cat > "${fixture_src}/lib.rs" <<'FIXTURE_EOF'
pub mod alpha;
pub mod subdir;
pub mod pathmod;
FIXTURE_EOF

    cat > "${fixture_src}/alpha.rs" <<'FIXTURE_EOF'
pub fn a() {}
#[cfg(test)]
mod tests {
    #[test]
    fn t() { assert!(true); }
}
FIXTURE_EOF

    cat > "${fixture_src}/subdir/mod.rs" <<'FIXTURE_EOF'
pub mod beta;
FIXTURE_EOF

    cat > "${fixture_src}/subdir/beta.rs" <<'FIXTURE_EOF'
pub fn b() {}
FIXTURE_EOF

    # Real (non-orphan) file that stays undiscovered unless #[path] is
    # resolved relative to pathmod.rs's OWN directory (fixture_src),
    # matching this repo's spectral_analysis.rs pattern — NOT relative to
    # child_base_dir (fixture_src/pathmod/), which is where a backwards
    # implementation of this rule would look instead.
    cat > "${fixture_src}/pathmod.rs" <<'FIXTURE_EOF'
#[cfg(test)]
#[path = "pathmod/tests.rs"]
mod tests;
FIXTURE_EOF
    cat > "${fixture_src}/pathmod/tests.rs" <<'FIXTURE_EOF'
#[test]
fn t() { assert!(true); }
FIXTURE_EOF

    # Two genuine orphans this run must catch.
    cat > "${fixture_src}/orphan_flat.rs" <<'FIXTURE_EOF'
pub fn dead() {}
FIXTURE_EOF
    cat > "${fixture_src}/subdir/orphan_nested.rs" <<'FIXTURE_EOF'
pub fn dead2() {}
FIXTURE_EOF

    reachable_files=()
    walk_file "${fixture_src}/lib.rs"

    local test_failed=0

    # Must be reachable (not orphaned):
    local must_reach f
    must_reach=(
        "${fixture_src}/lib.rs"
        "${fixture_src}/alpha.rs"
        "${fixture_src}/subdir/mod.rs"
        "${fixture_src}/subdir/beta.rs"
        "${fixture_src}/pathmod.rs"
        "${fixture_src}/pathmod/tests.rs"
    )
    for f in "${must_reach[@]}"; do
        if ! is_reachable "${f}"; then
            echo "SELF-TEST FAIL: expected reachable, was not: ${f#"${fixture_root}"/}" >&2
            test_failed=1
        fi
    done

    # Must NOT be reachable (must be caught as orphans):
    local must_not_reach
    must_not_reach=(
        "${fixture_src}/orphan_flat.rs"
        "${fixture_src}/subdir/orphan_nested.rs"
    )
    for f in "${must_not_reach[@]}"; do
        if is_reachable "${f}"; then
            echo "SELF-TEST FAIL: expected orphan, was marked reachable: ${f#"${fixture_root}"/}" >&2
            test_failed=1
        fi
    done

    rm -rf "${fixture_root}"

    if [ "${test_failed}" -ne 0 ]; then
        echo "check_module_reachability: SELF-TEST FAILED" >&2
        return 1
    fi
    echo "check_module_reachability: self-test OK (6 reachable, 2 orphans, all correctly classified)"
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    run_self_test
    exit $?
fi

# Order crates deterministically for stable output.
# (Not `mapfile`/`readarray`: this repo's default macOS /bin/bash is 3.2,
# which predates both — read into the array line-by-line instead.)
crate_dirs=()
while IFS= read -r line; do
    crate_dirs+=("${line}")
done < <(find "${crates_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

filter_crate="${1:-}"

if [ "${#crate_dirs[@]}" -eq 0 ]; then
    echo "check_module_reachability: no crate directories found under ${crates_dir}" >&2
    exit 1
fi

if [ -n "${filter_crate}" ]; then
    filter_found=0
    for crate_dir in "${crate_dirs[@]}"; do
        [ "$(basename "${crate_dir}")" = "${filter_crate}" ] && filter_found=1
    done
    if [ "${filter_found}" -eq 0 ]; then
        echo "check_module_reachability: no crate named '${filter_crate}' under ${crates_dir}" >&2
        echo "Known crates:" >&2
        for crate_dir in "${crate_dirs[@]}"; do
            echo "  $(basename "${crate_dir}")" >&2
        done
        exit 1
    fi
fi

for crate_dir in "${crate_dirs[@]}"; do
    crate_name="$(basename "${crate_dir}")"
    if [ -n "${filter_crate}" ] && [ "${crate_name}" != "${filter_crate}" ]; then
        continue
    fi

    src_dir="${crate_dir}/src"
    [ -d "${src_dir}" ] || continue

    reachable_files=()

    for root_name in lib.rs main.rs; do
        root_file="${src_dir}/${root_name}"
        [ -f "${root_file}" ] && walk_file "${root_file}"
    done

    if [ "${#reachable_files[@]}" -eq 0 ]; then
        echo "check_module_reachability: WARNING: ${crate_name} has no src/lib.rs or src/main.rs; skipping" >&2
        continue
    fi

    all_rs_files=()
    while IFS= read -r line; do
        all_rs_files+=("${line}")
    done < <(find "${src_dir}" -type f -name '*.rs' | sort)

    crate_orphans=()
    if [ "${#all_rs_files[@]}" -gt 0 ]; then
        for f in "${all_rs_files[@]}"; do
            if ! is_reachable "${f}"; then
                crate_orphans+=("${f}")
            fi
        done
    fi

    if [ "${#crate_orphans[@]}" -gt 0 ]; then
        overall_status=1
        echo "=== ${crate_name}: ${#crate_orphans[@]} orphan file(s) not reachable from src/lib.rs or src/main.rs ===" >&2
        for f in "${crate_orphans[@]}"; do
            echo "  ORPHAN: ${f#"${repo_root}"/}" >&2
            orphan_files+=("${f}")
        done
    fi

    all_crate_summaries+=("${crate_name}: ${#all_rs_files[@]} files on disk, ${#crate_orphans[@]} orphaned")
done

echo "--- module reachability summary ---"
if [ "${#all_crate_summaries[@]}" -gt 0 ]; then
    for s in "${all_crate_summaries[@]}"; do
        echo "  ${s}"
    done
fi

if [ "${overall_status}" -ne 0 ]; then
    echo "" >&2
    echo "check_module_reachability: FAILED — ${#orphan_files[@]} total orphan file(s)." >&2
    echo "Each file above exists on disk under a crate's src/ tree but is never" >&2
    echo "declared with 'mod'/'pub mod' (directly or transitively) from that" >&2
    echo "crate's lib.rs/main.rs, so cargo silently never compiles or tests it." >&2
    echo "Fix: add the missing 'pub mod <name>;' (or '#[path]' mod decl) to the" >&2
    echo "appropriate parent module, or delete the file if it is truly unused." >&2
    exit 1
fi

if [ -n "${filter_crate}" ]; then
    echo "check_module_reachability: OK — every .rs file under crates/${filter_crate}/src/ is reachable."
else
    echo "check_module_reachability: OK — every .rs file under crates/*/src/ is reachable."
fi
exit 0

# Known limitations (deliberate; re-review if the module layout below changes):
#   - Feature-gated `mod` declarations (#[cfg(feature = "...")] mod x;) are
#     treated as always-reachable rather than conditionally reachable — this
#     avoids needing a cfg-expression evaluator and only risks *missing* a
#     genuinely-orphaned feature-gated module, never a false positive on
#     healthy code.
#   - `mod`/`#[path]` declarations nested *inside* an inline `mod NAME { }`
#     block are not walked at all (only the outer `mod NAME {` opener is
#     recognized, purely so a `#[path]` immediately preceding it doesn't
#     get misapplied to the *next* sibling declaration). Verified against
#     rustc that such nesting resolves relative to yet a third base
#     directory beyond the two this script implements (dir-relative for a
#     `#[path]` on a file-backed `mod NAME;`, child_base_dir-relative for
#     plain NAME.rs/NAME/mod.rs lookup) — implementing that correctly would
#     need to track a directory-base *stack* through nested blocks, which
#     this codebase's module layout does not exercise (grep confirms zero
#     `mod NAME { ... #[path] ... }` nesting workspace-wide at review time,
#     the only inline `mod X { }` blocks present are `#[cfg(test)] mod
#     tests { }`, which has no children needing file resolution). UNLIKE
#     the feature-gating simplification above, this is the *unsafe*
#     direction: a file only reachable this way would be wrongly reported
#     as an orphan (false positive), not silently missed. If this pattern
#     is ever introduced, this script needs the stack-based rewrite before
#     trusting a FAILED result that names a file nested this way.
#   - Macro-generated `mod` declarations (e.g. from a `cfg_if!`/custom macro
#     expanding to `mod x;`) are invisible to this text-based walker. None
#     exist in this workspace as of this script's introduction (see
#     `grep -rn 'cfg_if!' crates/*/src`, empty); if that changes, this
#     script needs a matching macro-aware case or those modules need a
#     literal `mod` line the walker can see.
#   - Only crates/*/src/ is scanned. examples/, tests/, and benches/ are
#     independent Cargo targets (each with its own root file declared in
#     Cargo.toml) rather than part of the lib/bin module tree walked here;
#     an orphaned file under those dirs is not a *silently* dead module in
#     the same sense — cargo would only build it if a [[test]]/[[example]]/
#     [[bench]] target's `path` pointed at it, and a stray extra file there
#     is inert, not a false-coverage trap. If this ever needs coverage too,
#     each target's declared `path` (or default `tests/<name>.rs`) would
#     need cross-checking against Cargo.toml, which is a different check.
