#!/usr/bin/env bash
#
# sonar-evidence.sh — capture SonarQube UI screenshots into docs/evidence/.
#
# SonarQube is a web app, so this uses Playwright (chromium) to log in and
# screenshot the project dashboard, issues, and measures. The server URL and
# admin password are read from the local credential store (KWallet); nothing
# sensitive is stored in the repo.
#
# The PNGs show the internal server's project view, so they are CUI: they are
# written to docs/evidence/ which is gitignored for images. Only the text gate
# report (docs/evidence/sonar-report.txt) is committed.

set -euo pipefail
cd "$(dirname "$0")/.."

secret() {
  if command -v secret-tool >/dev/null 2>&1; then
    secret-tool lookup "$@" 2>/dev/null || true
  elif command -v distrobox-host-exec >/dev/null 2>&1; then
    distrobox-host-exec secret-tool lookup "$@" 2>/dev/null || true
  fi
}

export SONAR_HOST_URL="${SONAR_HOST_URL:-$(secret service sonarqube item host-url)}"
export SONAR_ADMIN_PASSWORD="${SONAR_ADMIN_PASSWORD:-$(secret service sonarqube item admin-password)}"
export SONAR_PROJECT_KEY="${SONAR_PROJECT_KEY:-nitrogen-monitor-server-bazzite}"
# Where Playwright is installed (shared runner cache by default).
export PLAYWRIGHT_REQUIRE_BASE="${PLAYWRIGHT_REQUIRE_BASE:-$HOME/.cache/playwright-runner}"

: "${SONAR_HOST_URL:?not set and not in keyring (service=sonarqube item=host-url)}"
: "${SONAR_ADMIN_PASSWORD:?not set and not in keyring (service=sonarqube item=admin-password)}"

node scripts/capture-sonar-evidence.mjs
