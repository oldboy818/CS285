#! /bin/bash
set -euo pipefail

# Launch TensorBoard on the remote instance, forward the port to localhost,
# keep the SSH tunnel open for a while, then shut the VM down cleanly.

ZONE="us-west4-a"
INSTANCE_NAME="cs285-hw5"
FORWARD_PORT="6006"
TB_LOGDIR="~/data"
SLEEP_MINUTES="10"   # how long to keep the tunnel open locally

echo "Starting instance ${INSTANCE_NAME} in ${ZONE}..."
gcloud compute instances start "${INSTANCE_NAME}" --zone="${ZONE}"

# Wait until SSH is actually ready
echo "Waiting for SSH to become available..."
for i in {1..30}; do
  if gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" -- -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no "echo ok" &>/dev/null; then
    echo "SSH is ready."
    break
  fi
  echo "  SSH not ready yet... retrying ($i/30)"
  sleep 5
  if [[ $i -eq 30 ]]; then
    echo "SSH did not become ready in time." >&2
    exit 1
  fi
done

# Start TensorBoard in a tmux session (no shutdown scheduled yet to avoid pam_nologin lockout)
echo "Starting TensorBoard in tmux on the VM..."
gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" --command "
  bash -lc '
    tmux kill-session -t tb 2>/dev/null || true
    tmux new -d -s tb \"tensorboard --logdir ${TB_LOGDIR} --port ${FORWARD_PORT} --host 0.0.0.0\"
  '
"

# Open a local SSH tunnel and, *inside this same SSH session*, schedule a delayed shutdown
# via a background process so that logins are not blocked immediately.
echo "Opening SSH tunnel localhost:${FORWARD_PORT} -> ${INSTANCE_NAME}:${FORWARD_PORT}"
echo "Visit: http://localhost:${FORWARD_PORT}"
gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" -- \
  -L ${FORWARD_PORT}:localhost:${FORWARD_PORT} \
  bash -lc "nohup bash -c 'sleep ${SLEEP_MINUTES}m; sudo shutdown now' >/dev/null 2>&1 & echo 'Tunnel up for ${SLEEP_MINUTES} minutes...'; sleep ${SLEEP_MINUTES}m" || true

# Best-effort stop in case shutdown didn’t happen yet
echo "Stopping instance (best effort)..."
gcloud compute instances stop "${INSTANCE_NAME}" --zone="${ZONE}" --quiet || true

echo "Done."
