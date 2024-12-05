#!/bin/bash
#SBATCH --job-name=fe
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/job-%j.out

cp /home/space/datasets-sqfs/FLIIDNIID/Data.sqfs /tmp/

apptainer run -B /tmp/Data.sqfs:/cluster:image-src=/ --nv fl_bench.sif python generate_data.py -d femnist -vr 0.0 -tr 0.0