export CUDA_VISIBLE_DEVICES=0,1,2,3
PDB_Dir="/usr/commondata/local_public/rep/train/pdb/"
alignment_dir="/usr/commondata/local_public/rep/train/a3m/"
template_mmcif_dir="/usr/commondata/public/CASP14_Datasets/datasets/pdb_mmcif/mmcif_files/"
output_dir="/root/AlphaFold-Pytorch/openfold/output_dir"
python3 train_openfold.py $PDB_Dir $alignment_dir $template_mmcif_dir $output_dir\
    2021-10-10\
    --precision 16\
    --gpus 4\
    --replace_sampler_ddp True\
    --seed 42\
    --deepspeed_config_path ./deepspeed_config.json
    # --resume_from_ckpt ckpt_dir/
    # in multi-gpu settings, the seed must be specified
    # --template_release_dates_cache_path mmcif_cache.json \ 