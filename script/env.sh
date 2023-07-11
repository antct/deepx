set -e
umask 000
export PYTHONPATH="${PYTHONPATH}:deepx"
export NCCL_IB_DISABLE=1
PORTS=($(ss -tln | tail -n +2 | awk '{print $4}' | awk -F ':' '{print $2}'))
for port in {10000..12345}; do
    if [[ ! " ${PORTS[*]} " =~ " ${port} " ]]; then
        TCP_PORT=$port
        break
    fi
done
export TCP_PORT
echo "tcp port: ${TCP_PORT}"
if ! command -v nvidia-smi &>/dev/null; then
    NUM_GPU=1
else
    NUM_GPU=$(nvidia-smi -L | wc -l)
fi
echo "gpu num: ${NUM_GPU}"
DEVICES=(${CUDA_VISIBLE_DEVICES//,/ })
DEVICES=${#DEVICES[@]}
if [[ $DEVICES -ne 0 ]]; then
    NUM_GPU=($([ $NUM_GPU -le $DEVICES ] && echo "$NUM_GPU" || echo "$DEVICES"))
fi
echo "cuda num: ${NUM_GPU}"
export NUM_GPU
export TOKENIZERS_PARALLELISM=false
function launch() {
    accelerate launch \
        --config_file yaml/ddp.yaml \
        --num_machines 1 \
        --num_processes ${NUM_GPU} \
        --main_process_port ${TCP_PORT} $@
}
export -f launch
function ddp-launch() {
    NUM_PROCESSES=$((${GPU_NUM} * ${WORLD_SIZE}))
    accelerate launch \
        --config_file yaml/ddp.yaml \
        --machine_rank ${RANK} \
        --num_machines ${WORLD_SIZE} \
        --num_processes ${NUM_PROCESSES} \
        --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} $@
}
export -f ddp-launch
function deepspeed-launch() {
    NUM_PROCESSES=$((${GPU_NUM} * ${WORLD_SIZE}))
    accelerate launch \
        --config_file yaml/deepspeed.yaml \
        --machine_rank ${RANK} \
        --num_machines ${WORLD_SIZE} \
        --num_processes ${NUM_PROCESSES} \
        --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} $@
}
export -f deepspeed-launch
