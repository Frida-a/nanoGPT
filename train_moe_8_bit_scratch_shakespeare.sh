#!/bin/bash

scratch_out_dir="out/richard/from-scratch" &&

python data/shakespeare/prepare.py &&

mkdir -p "$scratch_out_dir/out-moe-shakespeare-8-bit" &&

echo "Created dir $scratch_out_dir/out-moe-shakespeare-8-bit" &&

python train-moe.py config/train_moe_shakespeare_8_bit.py --compile=False --out_dir="$scratch_out_dir/out-moe-shakespeare-8-bit" &&

echo "done."