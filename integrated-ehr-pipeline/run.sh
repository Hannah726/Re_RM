#!/bin/bash
#PBS -N rawmed_test
#PBS -l select=1:ncpus=16:mem=256gb
#PBS -l walltime=6:00:00
#PBS -q normal
#PBS -j oe
#PBS -o /home/users/nus/e1582377/RawMed/logs/preprocess_6and24.log

cd /home/users/nus/e1582377/RawMed/integrated-ehr-pipeline
source ~/miniconda3/bin/activate rawmed

# 定义观测窗口数组
obs_array=(6 24)

for obs in "${obs_array[@]}"; do
    echo "--- Preprocessing: mimic iv (obs size = $obs) ---"
    
    if [ $obs -eq 6 ]; then
        MAX_EVT=200
        MAX_PAT_TOK_LEN=25600
    else
        MAX_EVT=400
        MAX_PAT_TOK_LEN=51200
    fi

    python main.py \
      --ehr mimiciv \
      --data ~/RawMed/data/raw/MIMIC-IV \
      --dest ~/scratch/RawMed/data/processed_${obs} \
      --ext .csv \
      --num_threads 32 \
      --readmission \
      --diagnosis \
      --seed "0,1,2" \
      --first_icu \
      --mortality \
      --long_term_mortality \
      --max_event_size ${MAX_EVT} \
      --max_event_token_len 128 \
      --max_patient_token_len ${MAX_PAT_TOK_LEN} \
      --obs_size ${obs} \
      --pred_size 24
done