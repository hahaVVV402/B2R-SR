Turn budget wrap-up was requested after 6 assistant turns (soft limit 6, grace 2). Process-mode live steering is unavailable, so the child was warned at launch to wrap up by this budget. Output may be partial.

## Review

- **Blocker:** `executed_source/run_featurize.sh:96-98` sets `BUNDLE_DONE=1` and `ARCHIVE_VERIFIED=1` before `sync` succeeds. If `sync` fails, `finalize` sees both flags set and can release the instance (`:151-165`) despite failed persistence. Set both flags only after successful `sync`.
- **High:** `executed_source/formal_recovery.py:1067-1092` validates per-image hashes/counts and model means, but only checks that `across_seeds` dataset keys exist. Arbitrary seed means/std/95% CIs would pass validation and enter a complete bundle, despite the frozen CI contract at `protocol.json:139`.
- **Note:** The bundle allowlist is bounded but not exact: all files beneath `source_snapshot`, matching filenames anywhere under `training`, and every evaluation `.jsonl` are included (`formal_recovery.py:1241-1259`). Stale unexpected files can therefore enter the archive.
- **Correct:** All five source-manifest SHA-256 values and byte sizes independently match.
- **Correct:** Completed runs bind goal/protocol/source manifest, scale, seed, run config, contiguous 500-step trace, and checkpoint hash (`formal_recovery.py:603-650`).
- **Correct:** Preflight exhaustively reads every recovery and evaluation pair (`formal_recovery.py:436-454, 681-696`); evaluation uses the frozen top-left modcrop contract (`:398-414`).
- **Correct:** Sibling archive SHA verification is portable (`run_featurize.sh:93-94`), signals are forwarded to the active workload (`:45-62`), and raw fallback archives cannot authorize release (`:102-115, 154-160`).
- **Correct:** Goal files are currently ignored and untracked, confirming selective `git add -f` remains necessary; no files are staged.

**NOT READY** for selective force-add/commit/push until the blocker and high finding are corrected.