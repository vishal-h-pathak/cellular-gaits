#!/usr/bin/env bash
#
# sentry-autostart-setup.sh — run ONCE on sentry, inside the WSL2 Ubuntu shell.
# Makes the box come back on the tailnet by itself after a reboot, so you never have to hand-start
# ssh / tailscaled / WSL again. Idempotent + non-destructive: safe to re-run.
#
# It does the WSL/Linux half automatically:
#   - ensures systemd is enabled in /etc/wsl.conf,
#   - enables + starts the ssh and tailscaled services on boot,
#   - makes sure tailscale is authed/up.
# It then PRINTS the Windows half (a Task Scheduler entry that boots WSL at logon) for you to paste
# into an elevated PowerShell on Windows — that part can't be done safely from inside WSL.

set -uo pipefail
say () { printf '\n>>> %s\n' "$1"; }

# --- sanity: are we actually inside WSL? ---
if ! grep -qiE 'microsoft|wsl' /proc/sys/kernel/osrelease 2>/dev/null; then
  echo "This script is meant to run INSIDE the WSL2 Ubuntu shell on sentry. Aborting."; exit 1
fi
DISTRO="${WSL_DISTRO_NAME:-Ubuntu}"
echo "Running inside WSL distro: $DISTRO"

# --- 1) systemd on (needed for enable-on-boot; usually already on) ---
say "ensuring systemd is enabled in /etc/wsl.conf"
if grep -qiE '^\s*systemd\s*=\s*true' /etc/wsl.conf 2>/dev/null; then
  echo "systemd already enabled."
  SYSTEMD_WAS_OFF=0
else
  sudo mkdir -p /etc
  if grep -q '^\[boot\]' /etc/wsl.conf 2>/dev/null; then
    sudo sed -i '/^\[boot\]/a systemd=true' /etc/wsl.conf
  else
    printf '\n[boot]\nsystemd=true\n' | sudo tee -a /etc/wsl.conf >/dev/null
  fi
  echo "Enabled systemd in /etc/wsl.conf (a 'wsl --shutdown' from Windows is needed for it to take effect)."
  SYSTEMD_WAS_OFF=1
fi

# --- 2) enable + start ssh and tailscaled on boot ---
say "enabling ssh + tailscaled to start on boot"
if pidof systemd >/dev/null 2>&1; then
  sudo systemctl enable --now ssh        || echo "WARN: could not enable ssh (is openssh-server installed? 'sudo apt install -y openssh-server')"
  sudo systemctl enable --now tailscaled || echo "WARN: could not enable tailscaled"
  echo "--- service status ---"
  systemctl is-enabled ssh tailscaled 2>/dev/null || true
  systemctl is-active  ssh tailscaled 2>/dev/null || true
else
  echo "systemd is not PID 1 yet (you just enabled it). After running 'wsl --shutdown' on Windows and"
  echo "reopening Ubuntu, re-run this script so 'systemctl enable' sticks."
fi

# --- 3) make sure tailscale is authed/up (state persists across reboots once done) ---
say "bringing tailscale up (no-op if already up)"
sudo tailscale up || echo "If this asked you to log in, complete it once; the auth persists afterward."
echo "--- tailscale ---"
tailscale ip -4 2>/dev/null || true
tailscale status 2>/dev/null | head -5 || true

# --- 4) the Windows half: boot WSL at logon so the services above actually come up ---
cat <<'PS'

================================================================================
WINDOWS HALF (do this once). WSL2 does NOT auto-boot at startup, so even with the
services enabled above, nothing runs until the distro is started. Add a Task that
boots WSL at logon and holds it open. In an ELEVATED PowerShell on Windows, paste:

  $act = New-ScheduledTaskAction -Execute "wsl.exe" `
         -Argument '-d Ubuntu -u root -- sh -c "while true; do sleep 3600; done"'
  $trg = New-ScheduledTaskTrigger -AtLogOn
  $set = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries `
         -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
  Register-ScheduledTask -TaskName "WSL keepalive (sentry compute)" `
         -Action $act -Trigger $trg -Settings $set -RunLevel Highest -Force

Notes:
  - If your distro isn't named "Ubuntu", run `wsl -l -v` and change -d Ubuntu to match.
  - This boots WSL when YOU log in. If the box reboots unattended (no login), enable
    Windows auto-login (netplwiz) — a security tradeoff; your call.
  - A brief console window may appear at logon; that's the keepalive holding the VM.
================================================================================
PS

if [ "${SYSTEMD_WAS_OFF:-0}" = "1" ]; then
  say "ACTION: systemd was just turned on. From Windows run 'wsl --shutdown', reopen Ubuntu, then re-run this script once."
fi
echo "Done. Verify from the Mac after a reboot:  tailscale ping 100.86.154.46 && ./ops/fleet-preflight.sh quick"
