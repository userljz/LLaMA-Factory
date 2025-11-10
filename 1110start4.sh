export HF_HOME='/wekafs/jinzeli2/cache'
export HF_HUB_OFFLINE='1'
export HIP_VISIBLE_DEVICES='4'
export WANDB_PROJECT="251109_llama_factory"
export WANDB_API_KEY="88b970302b89c7b55c90532cfd69ce4ee64ba81a"


# 基础配置文件
CONFIG_FILE="examples/train_lora/251109_llama3_1_8b_sft_wandb_eval.yaml"


# 要测试的 batch size 列表
# BATCH_SIZES=(64)
# lora_rank=(16)

# # 循环运行实验
# for batch_size in "${BATCH_SIZES[@]}"; do
#     for r in "${lora_rank[@]}"; do

#     exp_name=llama31_8b_sft_bs${batch_size}_rank${r}
#     echo "=========================================="
#     echo "Running experiment ${exp_name}"
#     echo "=========================================="
    
#     # 使用命令行参数覆盖配置，并设置不同的 output_dir
#     llamafactory-cli train ${CONFIG_FILE} \
#         per_device_train_batch_size=${batch_size} \
#         output_dir=saves/llama3.1-8b/sft/${exp_name} \
#         run_name=${exp_name} \
#         lora_rank=${r} \
#         &> logs/${exp_name}.log
    
#     echo "Experiment ${exp_name} completed!"
#     echo ""

#     done
# done

exp_name=251110_llama31_8b_sft_bs64_rank16_lr1e3_withEval
llamafactory-cli train ${CONFIG_FILE} \
        output_dir=saves/llama3.1-8b/sft/${exp_name} \
        run_name=${exp_name} \
        &> logs/${exp_name}.log