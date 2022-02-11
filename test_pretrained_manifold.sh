target_path="./tests/test_data/short.fasta"
database_path="/usr/commondata/public/CASP14_Datasets/datasets"
python3 run_pretrained_openfold.py \
    $target_path \
    $database_path/uniref90/uniref90.fasta \
    $database_path/mgnify/mgy_clusters.fa \
    $database_path/pdb70/pdb70 \
    $database_path/pdb_mmcif/mmcif_files/ \
    $database_path/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --use_precomputed_alignments ./output_dir/alignments \
    --output_dir ./output_dir \
    --bfd_database_path $database_path/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --model_device cuda:1 \
    --jackhmmer_binary_path /opt/anaconda3/envs/Manifold/bin/jackhmmer \
    --hhblits_binary_path /opt/anaconda3/envs/Manifold/bin/hhblits \
    --hhsearch_binary_path /opt/anaconda3/envs/Manifold/bin/hhsearch \
    --kalign_binary_path /opt/anaconda3/envs/Manifold/bin/kalign