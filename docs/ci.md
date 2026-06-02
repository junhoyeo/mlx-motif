# CI overview

## Hosted lane (`ci.yml`)

Runs on GitHub-hosted `macos-14` runners (Intel, Swift 6.0.x / Xcode 16).

| Job | What it checks |
|-----|---------------|
| `lint` | Python ruff lint + format (ubuntu-latest) |
| `test` | Python pytest, Python 3.11 + 3.12 (macos-14) |
| `swift` | Default Swift package (no MLX), `MotifChatApp` packaging (macos-14 / Swift 6.0.x) |

**Limitations:** the hosted runner has no Apple Silicon GPU, no Xcode 26 SDK, and
no Metal toolchain capable of compiling `mlx.metallib`. As a result the following
surfaces are **never exercised** by this lane:

- `MOTIFKIT_ENABLE_MLX=1` — `MotifKitMLX`, `MotifNativeGenerate/Evaluate/Serve`,
  and `MotifDecodeBench` are not built.
- `MOTIFKIT_RUN_MLX_RUNTIME_TESTS=1` — `MotifMetalKernelsTests` GPU dispatch tests
  are skipped.
- `#if compiler(>=6.2)` — the Liquid Glass `GlassStyle.swift` branch (`glassEffect`,
  `GlassEffectContainer`, `.buttonStyle(.glass)`) is dead code on Swift 6.0.x and
  is never compiled.

Several native bugs slipped through CI because none of these paths were reachable
from a hosted runner.

---

## Apple Silicon lane (`ci-mlx.yml`)

Runs on a **self-hosted** Apple Silicon runner with labels `[self-hosted, macOS, ARM64]`.

This lane covers everything the hosted lane cannot:

| Step | What it exercises |
|------|------------------|
| Build MLX overlay | `MotifKitMLX` + all MLX-gated targets (resolves mlx-swift deps) |
| Build `MotifDecodeBench` | native decode-throughput benchmark binary |
| Build `mlx.metallib` | Metal compute library required by runtime GPU tests |
| Run tests with `MOTIFKIT_RUN_MLX_RUNTIME_TESTS=1` | `MotifMetalKernelsTests` GPU dispatch, fused-vs-unfused QKV parity, Q4 decode correctness |
| Swift 6.2+ toolchain | `#if compiler(>=6.2)` Liquid Glass branch in `GlassStyle.swift` |

### Non-blocking contract

`ci-mlx.yml` is a **separate workflow file** — it is never added as a required
status check on PRs. If no self-hosted runner is online, the job queues silently
and does not block merging. The hosted lane (`ci.yml`) remains the gate.

---

## Registering a self-hosted macOS 26 / Xcode 26 runner

### Hardware requirements

- Apple Silicon Mac (M1 or later)
- macOS 26 (Tahoe) or later
- At least 16 GB RAM (mlx-swift resolution + Metal compilation is memory-intensive)
- At least 40 GB free disk (SwiftPM caches + Metal intermediates)

### Software requirements

1. **Xcode 26** installed at `/Applications/Xcode_26.app` (or any path matching
   `Xcode_26*.app`).  The workflow's toolchain-selection step scans for
   `version 6.[2-9]` or later in `swift --version`.
2. Select it as the active toolchain:
   ```sh
   sudo xcode-select -s /Applications/Xcode_26.app/Contents/Developer
   swift --version   # should print "Swift version 6.2.x" or later
   ```
3. Accept the Xcode licence:
   ```sh
   sudo xcodebuild -license accept
   ```
4. Verify `xcrun metal` works (needed by `build_mlx_swift_metallib.sh`):
   ```sh
   xcrun -sdk macosx metal --version
   ```

### Runner registration

1. In the GitHub repo go to **Settings → Actions → Runners → New self-hosted runner**.
2. Select **macOS** / **ARM64**.
3. Follow the on-screen download + configure steps.  When prompted for **labels**,
   enter exactly:
   ```
   self-hosted,macOS,ARM64
   ```
   (these three labels are what `ci-mlx.yml` uses in `runs-on`).
4. Install the runner as a `launchd` service so it survives reboots:
   ```sh
   ./svc.sh install
   ./svc.sh start
   ```

### Verifying the runner works

After registration, trigger the workflow manually:

```sh
gh workflow run ci-mlx.yml
```

A passing run confirms that mlx-swift resolves, `mlx.metallib` compiles, and the
GPU runtime tests execute correctly on the machine.

### Security note

Self-hosted runners execute arbitrary code from pull requests.  For a public repo,
restrict the runner to **trusted contributors only** under
**Settings → Actions → General → Fork pull request workflows** — set it to
"Require approval for all outside collaborators".
