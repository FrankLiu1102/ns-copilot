#!/bin/bash
# NS-Copilot Docker Manager
#
# Daily development:
#   ./run.sh                    — Start service
#   ./run.sh restart            — Restart container (after code changes, 5-10s)
#   ./run.sh logs               — View live logs
#   ./run.sh stop               — Stop service
#
# Build commands (first run or dependency changes):
#   ./run.sh build              — Rebuild image (~15-20min for flash-attn)
#   ./run.sh rebuild            — Full rebuild (no cache)
#
# Other:
#   ./run.sh status             — Check container status
#   ./run.sh shell              — Enter container shell

cd "$(dirname "$0")"

case "${1:-start}" in
  start|"")
    echo "Starting NS-Copilot..."
    docker-compose up -d
    echo ""
    echo "Service started!"
    echo "Visit: http://localhost:8001"
    echo "Logs:  ./run.sh logs"
    ;;

  restart)
    echo "Restarting container..."
    docker-compose restart
    echo ""
    echo "Container restarted (5-10s)"
    echo "Visit: http://localhost:8001"
    ;;

  stop)
    echo "Stopping service..."
    docker-compose down --remove-orphans
    echo "Stopped."
    ;;

  build)
    echo "Rebuilding image..."
    echo "This will take ~15-20min (flash-attn compilation)"
    docker-compose down --remove-orphans 2>/dev/null
    docker rm -f $(docker ps -aq --filter "name=neuro-copilot") 2>/dev/null
    docker-compose up -d --build
    echo ""
    echo "Build complete, service started!"
    echo "Visit: http://localhost:8001"
    ;;

  rebuild)
    echo "Full rebuild (no cache)..."
    echo "This will take ~15-20min"
    docker-compose down --remove-orphans 2>/dev/null
    docker rm -f $(docker ps -aq --filter "name=neuro-copilot") 2>/dev/null
    docker-compose build --no-cache && docker-compose up -d
    echo ""
    echo "Full rebuild complete, service started!"
    echo "Visit: http://localhost:8001"
    ;;

  logs)
    echo "Live logs (Ctrl+C to exit)..."
    docker-compose logs -f
    ;;

  status)
    echo "Container status:"
    docker-compose ps
    echo ""
    echo "Image info:"
    docker images | grep neuro
    ;;

  shell)
    echo "Entering container shell..."
    docker-compose exec neuro-copilot-web /bin/bash
    ;;

  help|--help|-h)
    echo "NS-Copilot Docker Manager"
    echo ""
    echo "Daily development:"
    echo "  ./run.sh                    — Start service"
    echo "  ./run.sh restart            — Restart container (after code changes, 5-10s)"
    echo "  ./run.sh logs               — View live logs"
    echo "  ./run.sh stop               — Stop service"
    echo ""
    echo "Build commands (first run or dependency changes):"
    echo "  ./run.sh build              — Rebuild image (~15-20min)"
    echo "  ./run.sh rebuild            — Full rebuild (no cache)"
    echo ""
    echo "Other:"
    echo "  ./run.sh status             — Check container status"
    echo "  ./run.sh shell              — Enter container shell"
    echo "  ./run.sh help               — Show this help"
    echo ""
    echo "Tip: After modifying Python code, just './run.sh restart' (5-10s)."
    echo "     Only rebuild when adding dependencies or modifying Dockerfile."
    ;;

  *)
    echo "Unknown command: $1"
    echo "Run './run.sh help' for usage."
    exit 1
    ;;
esac
