  #!/bin/bash
  # Launch all 6 experiments with 2-minute intervals                                                                         
                  
  set -euo pipefail

  JOBS_DIR="$(dirname "$0")/jobs"
  LOG_DIR="/workspace/RAGEN/logs/search_benchmark"
  mkdir -p "$LOG_DIR"

  SCRIPTS=(
      gpu0_qwen3b_ppo_filter.sh
      gpu1_qwen3b_ppo_nofilter.sh
      gpu2_qwen3b_grpo_filter.sh
      gpu3_qwen3b_grpo_nofilter.sh
      gpu4_llama3b_ppo_filter.sh
      gpu5_llama3b_ppo_nofilter.sh
  )

  PIDS=()
  for i in "${!SCRIPTS[@]}"; do
      script="${SCRIPTS[$i]}"
      name="${script%.sh}"
      log="${LOG_DIR}/${name}.log"

      echo "[$(date '+%H:%M:%S')] Starting ${script} → ${log}"
      bash "${JOBS_DIR}/${script}" > "$log" 2>&1 &
      PIDS+=($!)

      if [ $i -lt $((${#SCRIPTS[@]} - 1)) ]; then
          echo "[$(date '+%H:%M:%S')] Waiting 2 minutes before next..."
          sleep 120
      fi
  done

  echo ""
  echo "[$(date '+%H:%M:%S')] All 6 experiments launched. PIDs: ${PIDS[*]}"
  echo "Monitor with: tail -f ${LOG_DIR}/gpu*.log"
  echo "GPU usage:    watch nvidia-smi"

  for i in "${!PIDS[@]}"; do
      wait "${PIDS[$i]}" && echo "${SCRIPTS[$i]}: SUCCESS" || echo "${SCRIPTS[$i]}: FAILED"
  done
  SCRIPT
  chmod +x /workspace/RAGEN/scripts/vast/launch_all.sh
