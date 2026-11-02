#!/usr/bin/env bash
# ==============================================================================
# AETHER-VISION: AI Classroom Surveillance & Tactical Monitoring Launch Script
# ==============================================================================

set -e
PORT=${PORT:-5000}

# 1. Auto-free port if a zombie or previous instance was left running
if command -v fuser >/dev/null 2>&1; then
    PID=$(fuser $PORT/tcp 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "⚠️  Freeing lingering process on port $PORT..."
        fuser -k $PORT/tcp 2>/dev/null || true
        sleep 1
    fi
fi

echo "=================================================================="
echo "⚡ INITIALIZING AETHER-VISION NEURAL CLASSROOM MONITORING"
echo "=================================================================="

# Check Python environment
python3 -c "import cv2, flask, ultralytics; print('All core computer vision and deep learning packages verified.')"

echo ""
echo "🔥 Starting Tactical Surveillance Server on port $PORT..."
echo "🌐 URL: http://localhost:$PORT"
echo "=================================================================="

# 2. Start server in background
python3 app.py &
SERVER_PID=$!

# Trap cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down AETHER-VISION Surveillance Engine..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 3. Wait for server to become responsive
echo "⏳ Waiting for neural vision engine to warm up..."
for i in {1..20}; do
    if curl -s "http://127.0.0.1:$PORT/api/telemetry" >/dev/null 2>&1; then
        echo "✅ Surveillance server active!"
        break
    fi
    sleep 0.5
done

# 4. Auto-launch GUI in dedicated native app window
URL="http://localhost:$PORT"
echo "🖥️  Launching Tactical Command Center GUI..."

if command -v chromium >/dev/null 2>&1; then
    echo "🚀 Spawning dedicated native HUD window via Chromium..."
    chromium --app="$URL" >/dev/null 2>&1 &
elif command -v brave >/dev/null 2>&1; then
    echo "🚀 Spawning dedicated native HUD window via Brave..."
    brave --app="$URL" >/dev/null 2>&1 &
elif command -v zen-browser >/dev/null 2>&1; then
    echo "🚀 Opening in Zen Browser..."
    zen-browser "$URL" >/dev/null 2>&1 &
elif command -v firefox >/dev/null 2>&1; then
    echo "🚀 Opening in Firefox..."
    firefox "$URL" >/dev/null 2>&1 &
elif command -v xdg-open >/dev/null 2>&1; then
    echo "🚀 Opening in default system browser..."
    xdg-open "$URL" >/dev/null 2>&1 &
fi

echo ""
echo "=================================================================="
echo "🎯 MISSION CONTROL IS LIVE!"
echo "Press Ctrl+C to terminate session."
echo "=================================================================="

# Wait for server process
wait $SERVER_PID
