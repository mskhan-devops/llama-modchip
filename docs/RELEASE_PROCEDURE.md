# Mandatory Release Procedure (`synchopate/llama-modchip`)

**Policy:** NO `git tag vX.Y.Z` and NO `gh release create` may be executed until **all 7 Hard Gates** below pass with exit code `0`. Bypassing any gate is prohibited.

---

## Gate 0 — Release Manifest & Version Freeze
1. Create a release branch `release/vX.Y.Z` from clean `main`.
2. Update [`release_manifest.json`](../release_manifest.json) **FIRST**:
   - `release_version`, `release_tag`, `patch_file`, `upstream_llama_cpp.{commit,build_number}`
   - `abi_enums` matching `ggml/include/ggml.h` (`41 = Q1_0`, `42 = Q2_0`, `43 = F8_E4M3`, `44 = FRACTAL_64`, `45 = VQ3_64`)
   - `benchmarks` JSON paths and expected rounded display metrics
3. Confirm every HuggingFace model preset in [`install.sh`](../install.sh) pins an immutable **40-character commit SHA** (`MODEL_REV="<40-hex>"`; floating `"main"` is forbidden).

---

## Gate 1 — Unified Patch, ABI Enum & Header Sync
1. Regenerate `patches/llama-modchip-vX.Y.patch` cleanly against upstream `ggml-org/llama.cpp @ <commit>` (verify `git apply --check` passes on a clean upstream checkout).
2. Archive superseded patches under `patches/archive/` (never leave two top-level `llama-modchip-v*.patch` files ambiguous).
3. Verify byte-for-byte identity between `speculative/cpp/{rc_arbiter_l1.h,rc_predictor.h,rc_routecrack2.h,rc_stockfish.h,rc_trie_v2.h}` and their corresponding `common/rc_*.h` copies inside `patches/llama-modchip-vX.Y.patch`.
4. Verify [`scripts/gguf_tensor_type.py`](../scripts/gguf_tensor_type.py) `GGML_TYPES` matches `ggml/include/ggml.h` inside the patch for **every** enum ID (`0..45+`).
5. Compute and record `sha256sum patches/llama-modchip-vX.Y.patch` in `release_manifest.json`, `README.md`, `RELEASE_NOTES.md`, and `./modchip` (`cmd_checksums`).

---

## Gate 2 — Build, Test & Sanitizer Matrix
1. **Shell & Python static validation:**
   - `bash -n` on every `.sh` file + `install.sh` + `modchip`.
   - `python3` AST compilation on every tracked `.py` file.
2. **Unit tests:**
   - `python3 -m unittest discover -s gateway -p "test_*.py" -v`
   - `python3 -m unittest discover -s kv_cache -p "test_*.py" -v`
3. **C++ RouteCrack 2.0 multi-ISA sanitizer test suite (mandatory):**
   - AVX-512 / native: `g++ -std=c++17 -O2 -Wall -Wextra -Werror -march=native -fsanitize=address,undefined -Ispeculative/cpp speculative/cpp/test_routecrack2.cpp -o /tmp/test_rc2_avx512 && /tmp/test_rc2_avx512`
   - AVX2+FMA: `g++ -std=c++17 -O2 -Wall -Wextra -Werror -mavx2 -mfma -mno-avx512f -fsanitize=address,undefined -Ispeculative/cpp speculative/cpp/test_routecrack2.cpp -o /tmp/test_rc2_avx2 && /tmp/test_rc2_avx2`
   - Portable Scalar: `g++ -std=c++17 -O2 -Wall -Wextra -Werror -mno-avx512f -mno-avx2 -fsanitize=address,undefined -Ispeculative/cpp speculative/cpp/test_routecrack2.cpp -o /tmp/test_rc2_scalar && /tmp/test_rc2_scalar`
   - Verify 100% IEEE-754 bit-identity across all ISAs on adversarial `NaN`/`±Inf` features (and `<= 1e-6` cross-ISA parity on general dense finite feature vectors) and `sizeof(RC2L1ArbiterWeights) == 832`.
4. **Multi-ISA binary verification:**
   - Verify `avx512` binary selection requires `avx512f + avx512_vbmi + avx512_vnni` across `install.sh`, `modchip`, `scripts/check_cpu_readiness.py`, and `README.md`.
   - Verify `avx2` portable binaries contain zero AVX-512 (`zmm`) instructions.
   - Verify `-DLLAMA_BUILD_NUMBER=<build_number>` is set in `scripts/build_zen5_llama.sh`, `fedora/nemotron.sh`, and `apple-silicon/build-dsv4-engine.sh`.

---

## Gate 3 — Cross-Document, Technical Paper (`paper/`), Platform & Template Sync
Update **every** surface in the Mandatory Sync Matrix before running Gate 4:

| Surface | Mandatory Checks / Updates |
|---|---|
| `README.md` | Version badges, Hero Quality-Gate link, Paper PDF link, Benchmark tables (synced to JSON), Leaderboard top rows, Footnotes (default `--spec-type`), Checksums section, Repo directory map |
| `RELEASE_NOTES.md` | New `vX.Y.Z` top section, complete binary/tool/tarball/patch SHA-256 table, exact JSON-derived benchmark table, upgrade notes |
| `ARCHITECTURE.md` | Top banner, zero-collision ABI IDs (`44`/`45`), zero deleted script/patch references, architecture sections for every active subsystem |
| `FAQ.md` | Default presets, CLI flags (`--spec-type`, `--all-tools`), VRAM/RAM tiers |
| `paper/` (`main.tex` -> `llama-modchip-paper.pdf` -> `arxiv-submission.tar.gz`) | Bump version string, include new algorithms/results, verify `>= 1` vector PDF figure, verify `\clearpage` before `\begin{thebibliography}` (no post-References floating tables), rebuild `llama-modchip-paper.pdf`, rebuild `arxiv-submission.tar.gz` from exact `main.tex` + `figures/*.pdf` |
| `benchmarks/` | Update Quality Gate Markdown linked by `README.md`; verify every `results/*_vXY.json` has non-null `model_label` & sanitized `model_path` |
| `docs/*` (`DEEPSEEK_V4_*`, `NEMOTRON_*`, `HARDWARE_*`, `WINDOWS_PORT.md`) | Sync ABI IDs (`44`/`45`), default speculator, valid CLI commands, remove deleted script/patch references |
| `install.sh`, `modchip`, `scripts/llama_modchip.py`, `scripts/check_cpu_readiness.py`, `scripts/chat.py`, `scripts/build_trunk_quant.sh` | Default tag/commit/build number, preset labels & Heretic/MaxQ metrics, universal SHA-256 verification on ALL presets, `--poll 0 --poll-batch 0`, `THP=madvise`, `modchip checksums` parity |
| `windows/*`, `apple-silicon/*`, `fedora/*`, `systemd/*` | Consistent English banners, matching `--spec-type`, `-ub`, `-ctk/-ctv`, `--poll 0 --poll-batch 0`, build number, non-hardcoded paths |
| `gateway/*`, `kv_cache/*`, `speculative/ROUTECRACK_TECHNIQUES.md` | Dynamic active model detection, per-UID `0600` `O_NOFOLLOW` `/dev/shm` paths, valid source symbol/line references |
| `.github/ISSUE_TEMPLATE/*` | Single canonical `cpu_benchmark_submission.yml` listing all active `vX.Y.Z` presets + `bug_report.yml` (no legacy duplicate benchmark forms) |

---

## Gate 4 — Benchmark Integrity, OPSEC & Automated Preflight Gate
Run the automated release gate locally in strict mode:
```bash
python3 scripts/preflight_release_check.py --strict --run-cpp-tests
```
**Must pass with `0` errors and `0` warnings** across all automated gates, including:
- Exact JSON <-> Markdown numeric reconciliation (`README.md`, `RELEASE_NOTES.md`, `benchmarks/QUALITY_GATE_VQ3_64.md`, `paper/main.tex`)
- Full OPSEC scan across all git-tracked files AND `pdftotext paper/llama-modchip-paper.pdf` (zero private paths, internal IPs, internal codenames, or secret tokens)
- Zero broken Markdown relative links
- Zero references to deleted scripts/patches or retired colliding ABI enums

---

## Gate 5 — Dual-Mirror Artifact Publish & SHA-256 Lock (GitHub Release + HF Mirror)
1. Upload all prebuilt portable Linux binaries (`llama-server`, `llama-cli`, `llama-bench`, `llama-perplexity`, `llama-quantize` for `avx512`, `avx2`, and generic fallback) to **both**:
   - **Primary:** GitHub Release draft `vX.Y.Z` (`https://github.com/synchopate/llama-modchip/releases/download/vX.Y.Z`)
   - **Fallback:** HuggingFace binary mirror (`https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF/resolve/main`)
   - Upload release bundle tarballs (e.g. `llama-modchip-vX.Y.Z-linux-x86_64.tar.gz`) to GitHub Release `vX.Y.Z` and record under `github_only_artifacts` in `release_manifest.json`.
2. Update `release_manifest.json`, `README.md`, `RELEASE_NOTES.md`, `install.sh`, and `modchip` with exact final SHA-256s.
3. Run remote dual-mirror verification:
   ```bash
   python3 scripts/preflight_release_check.py --strict --verify-remote
   ```
   Confirms HTTP `200` + exact matching `Content-Length` across **both** GitHub Release and HuggingFace mirror URLs for every dual-mirror binary (plus HF LFS SHA-256 header parity when exposed), exact SHA-256 parity of remote `SHA256SUMS` and `llama-modchip-vX.Y.patch` against `release_manifest.json`, and HTTP `200` on GitHub for release tarballs.

---

## Gate 6 — Clean-Room Smoke Test, Tag & Sign-Off
1. Clean-room Linux verification (fresh temporary workspace):
   - `./modchip download-bin --all-tools` -> SHA-256 verification passes for all downloaded tools
   - `./modchip checksums` -> matches `release_manifest.json`
   - `python3 scripts/check_cpu_readiness.py --no-bench` -> clean exit, accurate ISA (`avx512f + avx512_vbmi + avx512_vnni`) & preset tiers
   - `python3 scripts/llama_modchip.py --model /path/to/model.gguf --print-cmd` -> emits expected `--spec-type`, `-ub`, `--poll 0 --poll-batch 0`
2. Merge PR after GitHub Actions [`preflight.yml`](../.github/workflows/preflight.yml) is green.
3. Tag & publish:
   ```bash
   git tag -a vX.Y.Z -m "llama-modchip vX.Y.Z"
   git push origin vX.Y.Z
   gh release edit vX.Y.Z --draft=false
   ```
