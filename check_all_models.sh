#!/bin/bash
# Quick status check for all running models

echo "=============================================="
echo "FingerPrint Evaluation Status"
echo "=============================================="
echo ""
date
echo ""

echo "=== Screen Sessions ==="
screen -ls
echo ""

echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
  awk -F', ' '{printf "GPU %s: %s util, %s/%s VRAM, %s°C\n", $1, $3, $4, $5, $6}'
echo ""

echo "=== Model Progress ==="
for log in results/single_runs_35k/*.log results/multi_model_35k_*/*.log; do
    if [ -f "$log" ]; then
        model=$(basename "$log" .log)
        # Get last progress line
        progress=$(grep -o '[0-9]*%' "$log" | tail -1)
        if [ -n "$progress" ]; then
            echo "$model: $progress"
        fi
    fi
done
echo ""

echo "=== Database Results ==="
for db in results/single_runs_35k/*.db results/multi_model_35k_*/*.db; do
    if [ -f "$db" ]; then
        db_name=$(basename "$db")
        count=$(sqlite3 "$db" "SELECT COUNT(*) FROM probe_results;" 2>/dev/null || echo "0")
        images=$(sqlite3 "$db" "SELECT COUNT(DISTINCT image_id) FROM probe_results;" 2>/dev/null || echo "0")
        echo "$db_name: $count results ($images images)"
    fi
done
echo ""

echo "=============================================="
