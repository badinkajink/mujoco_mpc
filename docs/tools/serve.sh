#!/usr/bin/env bash
# Local static server for mujoco_mpc/docs (the experiment pages, media and
# videos), for viewing over VS Code / ssh port forwarding when artifact hosting
# is not available. Binds to 127.0.0.1 only; nothing is exposed on the LAN.
#
#   docs/tools/serve.sh start    # systemd --user unit golem-docs-http, survives logout (linger is on)
#   docs/tools/serve.sh stop
#   docs/tools/serve.sh status
#
# Then forward port 8765 (VS Code: Ports panel -> Forward a Port -> 8765) and open
#   http://localhost:8765/                                   directory listing
#   http://localhost:8765/lean/20260911-planner_ablation.html
#   http://localhost:8765/experiments/INDEX.md               (raw markdown)
set -euo pipefail
PORT="${PORT:-8765}"
DOCS="$(cd "$(dirname "$0")/.." && pwd)"
UNIT=golem-docs-http
case "${1:-status}" in
  start)
    if systemctl --user is-active --quiet "$UNIT"; then echo "already running: http://localhost:$PORT/"; exit 0; fi
    systemd-run --user --unit="$UNIT" --description="static server for $DOCS" \
      --property=Restart=on-failure --property=WorkingDirectory="$DOCS" \
      python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$DOCS" >/dev/null
    sleep 0.5
    systemctl --user is-active --quiet "$UNIT" && echo "serving $DOCS at http://localhost:$PORT/ (unit $UNIT)"
    ;;
  stop)   systemctl --user stop "$UNIT" 2>/dev/null && echo "stopped" || echo "not running" ;;
  status) systemctl --user status "$UNIT" --no-pager 2>/dev/null | head -5 || echo "not running"
          curl -s -o /dev/null -w "GET /lean/ -> HTTP %{http_code}\n" "http://127.0.0.1:$PORT/lean/" || true ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
