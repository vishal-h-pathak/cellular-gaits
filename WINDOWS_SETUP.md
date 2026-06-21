# Windows workstation setup (the compute half)

> **What this is.** Bare-metal-to-running bootstrap for the Windows box (5900X / 3080 Ti / 32 GB)
> so it can act as the **workstation** while the Mac is the **cockpit**. Tiered: do **Tier 0** to
> start a run tonight; **Tier 1** to let the Mac drive it remotely; **Tier 2** only when we reach
> the GPU/RL endgame. After Tier 1, day-to-day you launch runs from the Mac with `cockpit.sh` —
> see `../portfolio/cockpit.sh` and `../portfolio/docs/cellular-gaits/CROSS_MACHINE.md`.
>
> Pick your shell once: **WSL2 (Ubuntu) is recommended** for the compute environment — it gives a
> clean Linux shell, `tmux` for runs that survive disconnect, painless SSH, and the smooth path to
> the future JAX/MJX GPU work. Native Windows also works for interactive use; where steps differ
> they're marked **[WSL2]** / **[native]**.

---

## Tier 0 — minimum to start the nav run tonight (CPU only, no CUDA)

The navigation evolution is CPU-bound (classic MuJoCo); **you do not need CUDA or the GPU for it.**

1. **Git**
   - [native] Install Git for Windows (https://git-scm.com/download/win). Claude Code shells out
     to Git Bash, so this is required either way.
   - [WSL2] `wsl --install -d Ubuntu` in an admin PowerShell, reboot, set up your user, then
     `sudo apt update && sudo apt install -y git build-essential`.

2. **uv** (Python env/runner)
   - [native] PowerShell: `irm https://astral.sh/uv/install.ps1 | iex`
   - [WSL2] `curl -LsSf https://astral.sh/uv/install.sh | sh`

3. **Claude Code CLI** (for delegated agent sessions on the box)
   - [native] PowerShell: `irm https://claude.ai/install.ps1 | iex`
   - [WSL2] `curl -fsSL https://claude.ai/install.sh | bash` (then `claude` once to sign in)

4. **Clone both repos, side by side** (e.g. `~/dev/jarvis/` in WSL2, or `%USERPROFILE%\dev\jarvis\`)
   ```
   git clone https://github.com/vishal-h-pathak/cellular-gaits.git
   git clone https://github.com/vishal-h-pathak/portfolio.git
   cd cellular-gaits && git checkout feat/n-navigation && cd ..
   cd portfolio      && git checkout feat/n-navigation-scaffold && cd ..
   ```

5. **Bridge the chemo warm-start file** (gitignored in cellular-gaits; the committed copy is in
   portfolio). The nav run warm-starts from it — without this the run errors immediately.
   ```
   mkdir -p cellular-gaits/outputs/web_data_ch
   cp portfolio/public/cellular-gaits/data-ch/chemotaxis_controller.json cellular-gaits/outputs/web_data_ch/
   ```
   [native PowerShell] `Copy-Item portfolio\public\cellular-gaits\data-ch\chemotaxis_controller.json cellular-gaits\outputs\web_data_ch\`

6. **Env + start the run** (the command is also in `SYNC.md`):
   ```
   cd cellular-gaits
   uv sync
   uv run python scripts/run_evolution_navigation.py --pop 48 --gens 70 --checkpoint-every 5 --workers 16
   ```
   - To survive disconnect: [WSL2] run it inside `tmux` (`tmux new -s nav`, run, `Ctrl-b d` to
     detach, `tmux attach -t nav` to return). [native] keep the terminal open, or
     `Start-Process` it; simplest tonight is just leave the window open.
   - `--workers 16` leaves a few of the 5900X's 24 threads for the OS; lower it if the box stalls.

7. **Morning:** export the bundle + write `REPORT_n_a.md`, commit on `feat/n-navigation`, push,
   and update `../portfolio/docs/cellular-gaits/SYNC.md` (clear the claim, log the result).

---

## Tier 1 — let the Mac drive the box (remote cockpit)

1. **Tailscale** on the Windows box and the Mac, same account (https://tailscale.com/download).
   `tailscale ip -4` gives the box a stable `100.x` address; `tailscale status` lists peers.
2. **SSH server so the Mac can connect:**
   - [native] Settings → System → Optional features → Add → **OpenSSH Server**; then admin
     PowerShell: `Start-Service sshd; Set-Service -Name sshd -StartupType Automatic`.
   - [WSL2] `sudo apt install -y openssh-server tmux`, then `sudo service ssh start` (and enable on
     boot). Note WSL2's SSH listens inside the WSL VM — either run Tailscale inside WSL2, or add a
     `netsh interface portproxy` forward from the Windows host to the WSL IP. **Simplest: install
     Tailscale inside the WSL2 Ubuntu** so the box's `100.x` address *is* the Linux environment.
3. **Key auth Mac → box:** append your Mac's `~/.ssh/id_ed25519.pub` to the box's
   `~/.ssh/authorized_keys` ([native] `C:\Users\<you>\.ssh\authorized_keys`). Test from the Mac:
   `ssh <user>@<box-tailscale-name>`.
4. **Tell the cockpit who the box is:** on the Mac, set `WIN_HOST` (e.g.
   `export WIN_HOST="vishal@windows-box"` — the Tailscale MagicDNS name or `100.x` IP). Then
   `./cockpit.sh check` from the portfolio repo should print "reachable."

After this, you launch and monitor runs from the Mac — see `cockpit.sh` below.

---

## Tier 2 — GPU workstation (only for the RL / MJX endgame; not needed now)

The 3080 Ti does nothing for the current CPU CMA-ES. Do this when we start the connectome/RL work.

1. **NVIDIA driver** — install the latest Game Ready/Studio driver (likely already present).
   Verify: `nvidia-smi` shows the 3080 Ti and a CUDA version.
2. **PyTorch + CUDA (for RL):** with uv, add the CUDA build, e.g.
   `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`. PyTorch bundles its
   own CUDA runtime — **no separate CUDA Toolkit install needed** for Torch. Verify in Python:
   `import torch; torch.cuda.is_available()` → `True`.
   - [native] PyTorch-CUDA works on native Windows.
3. **JAX / MuJoCo-MJX (for GPU-vectorized physics):** JAX GPU on Windows requires **WSL2** (no
   native Windows CUDA wheels). In WSL2: `uv pip install "jax[cuda12]"` and the CUDA toolkit/cuDNN
   per the JAX install matrix; then MJX. This is the bigger lift — only worth it once we commit to
   the MJX port (contact-rich fly sim is MJX's hard case; verify FlyGym-on-MJX feasibility first).

---

## Notes

- Each machine runs its own `uv sync`; `.venv` is never synced. `uv.lock` **is** committed, so both
  machines resolve identical dependency versions (platform-specific wheels are handled by uv).
- Only small things cross git (code, docs, the web-export bundle); `checkpoints/`/`outputs/` stay
  local. Move big artifacts machine-to-machine over Tailscale (`scp`/`rsync`) or W&B, never git.
- Full protocol, roles, and the daily loop: `../portfolio/docs/cellular-gaits/CROSS_MACHINE.md`.
  Live state every session: `../portfolio/docs/cellular-gaits/SYNC.md`.
