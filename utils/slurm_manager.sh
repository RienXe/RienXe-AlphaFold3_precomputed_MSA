

set -euo pipefail

# ensure log exists
touch "$LOGFILE"

while true; do
  # get the last jobID from the log (empty if none)
  last_jid=$( awk 'END {print $2}' "$LOGFILE" || echo "" )

  # List all .sh scripts
  sh_scripts=$(find "$SCRIPT_DIR" -maxdepth 1 -type f -name "*.sh" | sort)

  if [[ -z "$sh_scripts" ]]; then
    echo "No .sh scripts found in $SCRIPT_DIR — sleeping for 2 minutes..."
    sleep 120  # 120 seconds = 2 minutes
    continue
  fi


  # loop through all .sh files in lexicographic order
  for script in $(printf '%s\n' "$SCRIPT_DIR"/*.sh | sort); do
    base=$(basename "$script")
    # skip if already in the log
    if grep -qx "$base" <(awk '{print $1}' "$LOGFILE"); then
      continue
    fi

    echo -n "Submitting $base"
    if [[ -n $last_jid ]]; then
      echo " (afterok:$last_jid)..."
      out=$( sbatch --dependency=afterany:"$last_jid" "$script" )
    else
      echo "..."
      out=$( sbatch "$script" )
    fi

    # parse new jobID and record it
    jid=$(echo "$out" | awk '{print $4}')
    echo "$base $jid" >> "$LOGFILE"
    last_jid=$jid
  done

  echo "Done scan — sleeping ${INTERVAL}s..."
  sleep "$INTERVAL"
done

