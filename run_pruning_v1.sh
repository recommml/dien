mkdir dnn_save_path
mkdir dnn_best_model
CUDA_VISIBLE_DEVICES=0  python script_pruning_v1/train.py train DIEN  >train_dein_prune.log 2>&1 &
