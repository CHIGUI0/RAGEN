#!/bin/bash
# Monitor retrieval server, auto-restart if down
LOG=/workspace/RAGEN/logs/retrieval_server_watchdog.log
echo "[$(date)] Watchdog started" >> "$LOG"

while true; do
    # Use long timeout (30s) to avoid killing busy-but-alive server
    # Also check if the process exists as a secondary signal
    if ! curl -s --max-time 30 http://127.0.0.1:8000/health > /dev/null 2>&1; then
        # Double-check: if server process is alive and responding to /retrieve, don't kill it
        if pgrep -f "retrieval/server.py" > /dev/null 2>&1; then
            echo "[$(date)] Health check timed out but server process alive, skipping restart" >> "$LOG"
        else
            echo "[$(date)] Server DOWN (process dead), restarting..." >> "$LOG"
            pkill -9 -f "retrieval/server.py" 2>/dev/null
            sleep 5
            eval "$(conda shell.bash hook)"
            conda activate ragen
            RETRIEVAL_ENCODER_DEVICE=cuda:2 OMP_NUM_THREADS=4 \
                nohup python scripts/retrieval/server.py \
                --data_dir ./search_data/prebuilt_indices \
                --port 8000 --host 127.0.0.1 >> logs/retrieval_server.log 2>&1 &
            echo "[$(date)] Restart issued (PID: $!), waiting 120s for load..." >> "$LOG"
            sleep 120
        fi
    else
        echo "[$(date)] Server OK" >> "$LOG"
    fi
    sleep 30
done
