export HF_HOME='/wekafs/jinzeli2/cache'
export HF_HUB_OFFLINE='1'
export HIP_VISIBLE_DEVICES='5'
export WANDB_PROJECT="251109_llama_factory"
export WANDB_API_KEY="88b970302b89c7b55c90532cfd69ce4ee64ba81a"


# 基础配置文件
CONFIG_FILE="examples/train_lora/251110_llama3_1_8b_sft_OnlyEval.yaml"


# 要测试的 batch size 列表
CKPT_NUM=(500 640)
# lora_rank=(8)

# 循环运行实验
for ckpt_num in "${CKPT_NUM[@]}"; do
    # for r in "${lora_rank[@]}"; do

    exp_name=251110_onlyEval_llama3.1-8b-lora-sft-accept-head-ckpt${ckpt_num}_tolerance01
    echo "=========================================="
    echo "Running experiment ${exp_name}"
    echo "=========================================="
    
    # 使用命令行参数覆盖配置，并设置不同的 output_dir
    llamafactory-cli train ${CONFIG_FILE} \
        adapter_name_or_path=/wekafs/jinzeli2/LLaMA-Factory/saves/llama3.1-8b/lora/sft-accept-head/checkpoint-${ckpt_num} \
        output_dir=saves/llama3.1-8b/only_eval/${exp_name} \
        run_name=${exp_name} \
        accept_head_accuracy_tolerance=0.1 \
        &> logs/${exp_name}.log
    
    echo "Experiment ${exp_name} completed!"
    echo ""

    # done
done

