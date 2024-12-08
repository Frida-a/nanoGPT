#!/bin/bash

scratch_out_dir="out/richard/from-scratch" &&


python data/shakespeare/prepare.py &&


mkdir -p "$scratch_out_dir/out-shakespeare-8-bit-gpt2-finetune" &&

python train_8_bit.py config/finetune_shakespeare.py --compile=False --out_dir="$scratch_out_dir/out-shakespeare-8-bit-gpt2-finetune" &&


echo "done."