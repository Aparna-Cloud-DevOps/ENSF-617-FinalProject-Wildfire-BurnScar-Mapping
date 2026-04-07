#!/bin/bash
# monitor.sh
# Auto-monitors the active training job, detects errors, and auto-fixes known ones.
# Run in a separate terminal: bash monitor.sh
# It will keep watching until the job finishes or all retries are exhausted.

set -e
PROJECT_DIR="/home/aparna.ayyalasomayaj/wildfire_burn_scar_mapping"
cd "$PROJECT_DIR"

MAX_RETRIES=3
POLL_INTERVAL=60   # seconds between log checks
retry_count=0

# ── helpers ──────────────────────────────────────────────────────────────────

get_active_job() {
    squeue -u aparna.ayyalasomayaj -h -o "%i %j %T" 2>/dev/null | head -1
}

get_job_id() {
    squeue -u aparna.ayyalasomayaj -h -o "%i" 2>/dev/null | head -1
}

get_latest_out() {
    ls -t "$PROJECT_DIR"/wf_train_all_*.out 2>/dev/null | head -1
}

get_latest_err() {
    ls -t "$PROJECT_DIR"/wf_train_all_*.err 2>/dev/null | head -1
}

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

# ── known error patterns and their auto-fixes ────────────────────────────────
# Each entry: "grep_pattern|fix_command"

declare -A FIXES
FIXES["unexpected keyword argument 'verbose'"]="
    log 'FIX: Removing verbose= from ReduceLROnPlateau in unet_training_v2.py and ablation.py'
    sed -i 's/, verbose=True//g; s/, verbose=False//g' unet_training_v2.py ablation.py
"
FIXES["CUDA out of memory"]="
    log 'FIX: Reducing batch size from 8 to 4 in pipeline_job1_train.slurm'
    sed -i 's/--batch      8/--batch      4/' pipeline_job1_train.slurm
"
FIXES["No module named"]="
    log 'FIX: Missing module detected — attempting pip install'
    MODULE=\$(grep -o \"No module named '[^']*'\" \"\$LATEST_OUT\" | head -1 | grep -o \"'[^']*'\" | tr -d \"'\")
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate gpuenv && pip install -q \"\$MODULE\"
"
FIXES["QOSMaxCpuPerUserLimit"]="
    log 'FIX: CPU quota exceeded — reducing cpus-per-task to 2'
    sed -i 's/--cpus-per-task=4/--cpus-per-task=2/' pipeline_job1_train.slurm pipeline_job2_eval.slurm
"
FIXES["out of memory"]="
    log 'FIX: RAM OOM — reducing batch size from 8 to 4'
    sed -i 's/--batch      8/--batch      4/' pipeline_job1_train.slurm
"

# ── resubmit after fix ────────────────────────────────────────────────────────

resubmit() {
    log "Resubmitting pipeline_job1_train.slurm (retry $((retry_count+1))/$MAX_RETRIES) ..."
    NEW_JOB=$(sbatch pipeline_job1_train.slurm | awk '{print $4}')
    log "New job ID: $NEW_JOB"
    retry_count=$((retry_count + 1))
    sleep 10
}

# ── main monitor loop ─────────────────────────────────────────────────────────

log "========================================="
log "  Wildfire Pipeline Auto-Monitor"
log "  Project: $PROJECT_DIR"
log "  Max retries: $MAX_RETRIES"
log "========================================="

while true; do
    JOB_INFO=$(get_active_job)
    JOB_ID=$(get_job_id)
    LATEST_OUT=$(get_latest_out)
    LATEST_ERR=$(get_latest_err)

    # ── Case 1: No job running ────────────────────────────────────────────────
    if [ -z "$JOB_ID" ]; then
        log "No active job found."

        # Check if the last job completed successfully
        if [ -f "$LATEST_OUT" ]; then
            if grep -q "Training complete" "$LATEST_OUT" 2>/dev/null || \
               grep -q "DONE  : TRAIN" "$LATEST_OUT" 2>/dev/null; then
                log "SUCCESS: Training completed. Submit pipeline_job2_eval.slurm next."
                log "  sbatch $PROJECT_DIR/pipeline_job2_eval.slurm"
                exit 0
            fi
        fi

        # Job ended without success — check for known errors
        if [ "$retry_count" -ge "$MAX_RETRIES" ]; then
            log "ERROR: Max retries ($MAX_RETRIES) reached. Manual intervention required."
            log "  Check: $LATEST_OUT"
            log "  Check: $LATEST_ERR"
            exit 1
        fi

        FIXED=false
        for PATTERN in "${!FIXES[@]}"; do
            if [ -f "$LATEST_OUT" ] && grep -q "$PATTERN" "$LATEST_OUT" 2>/dev/null; then
                log "DETECTED: $PATTERN"
                eval "${FIXES[$PATTERN]}"
                FIXED=true
                resubmit
                break
            fi
            if [ -f "$LATEST_ERR" ] && grep -q "$PATTERN" "$LATEST_ERR" 2>/dev/null; then
                log "DETECTED: $PATTERN"
                eval "${FIXES[$PATTERN]}"
                FIXED=true
                resubmit
                break
            fi
        done

        if [ "$FIXED" = false ]; then
            log "UNKNOWN ERROR — last 10 lines of .err:"
            tail -10 "$LATEST_ERR" 2>/dev/null || echo "  (no .err file)"
            log "Last 5 lines of .out:"
            tail -5 "$LATEST_OUT" 2>/dev/null || echo "  (no .out file)"
            if [ "$retry_count" -lt "$MAX_RETRIES" ]; then
                log "Attempting blind resubmit ..."
                resubmit
            else
                log "Manual fix required. Exiting."
                exit 1
            fi
        fi

    # ── Case 2: Job is pending ────────────────────────────────────────────────
    elif echo "$JOB_INFO" | grep -q "PD"; then
        REASON=$(squeue -u aparna.ayyalasomayaj -h -o "%R" 2>/dev/null | head -1)
        log "Job $JOB_ID is PENDING — reason: $REASON"

        # Auto-fix CPU quota while pending
        if echo "$REASON" | grep -q "QOSMaxCpuPerUserLimit"; then
            log "FIX: CPU quota — reducing cpus-per-task to 2 and requeuing"
            scancel "$JOB_ID"
            sed -i 's/--cpus-per-task=4/--cpus-per-task=2/g' pipeline_job1_train.slurm
            sleep 3
            resubmit
        fi

    # ── Case 3: Job is running ────────────────────────────────────────────────
    elif echo "$JOB_INFO" | grep -q " R "; then
        RUNTIME=$(squeue -u aparna.ayyalasomayaj -h -o "%M" 2>/dev/null | head -1)
        LAST_LINE=$(tail -1 "$LATEST_OUT" 2>/dev/null | sed 's/.*INFO *//')

        # Extract epoch info if available
        EP_LINE=$(grep "| Ep " "$LATEST_OUT" 2>/dev/null | tail -1 | sed 's/.*INFO *| *//')
        if [ -n "$EP_LINE" ]; then
            log "Job $JOB_ID RUNNING ($RUNTIME) | $EP_LINE"
        else
            log "Job $JOB_ID RUNNING ($RUNTIME) | $LAST_LINE"
        fi

        # Check for error keywords appearing in the running log
        for PATTERN in "${!FIXES[@]}"; do
            if grep -q "$PATTERN" "$LATEST_OUT" 2>/dev/null || \
               grep -q "$PATTERN" "$LATEST_ERR" 2>/dev/null; then
                log "ERROR DETECTED in running job: $PATTERN"
                log "Cancelling job $JOB_ID ..."
                scancel "$JOB_ID"
                sleep 5
                eval "${FIXES[$PATTERN]}"
                resubmit
                break
            fi
        done
    fi

    sleep "$POLL_INTERVAL"
done
