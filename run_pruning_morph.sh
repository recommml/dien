mkdir -p dnn_save_path_morph
mkdir -p dnn_best_model_morph
CUDA_VISIBLE_DEVICES=1  python script_pruning_morph/train.py train DIEN  >train_dein_prune_morph.log 2>&1 &
