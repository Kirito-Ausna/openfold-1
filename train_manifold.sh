export CUDA_VISIBLE_DEVICES=0,1,2,3
train_dir="/usr/commondata/local_public/rep/train/pdb/"
# train_dir="/usr/commondata/public/CASP14_Datasets/datasets/pdb_mmcif/mmcif_files/"
alignment_dir="/usr/commondata/local_public/rep/train/a3m/"
# alignment_dir="/root/AlphaFold-Pytorch/openfold/dataset_generate/alignment_dir/"
template_mmcif_dir="/usr/commondata/public/CASP14_Datasets/datasets/pdb_mmcif/mmcif_files/"
output_dir="/root/AlphaFold-Pytorch/openfold/output_dir"
python3 train_openfold.py $train_dir $alignment_dir $template_mmcif_dir $output_dir\
    2021-10-10\
    --precision 32\
    --gpus 4\
    --replace_sampler_ddp True\
    --seed 36\
    --deepspeed_config_path ./deepspeed_config.json\
    --max_epochs 10\
    # --wandb\
    --experiment_name Test1\
    --wandb_id kirito_asuna\
    --wandb_project Manifold_Exp4\
    --config_preset initial_training\


    # --resume_from_ckpt ckpt_dir/
    # in multi-gpu settings, the seed must be specified
    # --template_release_dates_cache_path mmcif_cache.json \ 