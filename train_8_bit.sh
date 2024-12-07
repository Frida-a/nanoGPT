#### Training quantized model from scratch with Int8

scratch_out_dir = "out/richard/from-scratch"

# Train on Shakespeare character-level dataset
python data/shakespeare_char/prepare.py

mkdir -p "$scratch_out_dir/out-shakespeare-char-8-bit"

python train_8_bit.py config/train_shakespeare_char_8_bit.py --compile=False --out_dir="$scratch_out_dir/out-shakespeare-char-8-bit"

# Train on Shakespeare word-level dataset
python data/shakespeare/prepare.py

mkdir -p "$scratch_out_dir/out-shakespeare-8-bit"

python train_8_bit.py config/train_shakespeare_8_bit.py --compile=False --out_dir="$scratch_out_dir/out-shakespeare-8-bit"

# From scratch on GPT2 with Shakespeare dataset
# mkdir -p "$scratch_out_dir/out-shakespeare-8-bit-gpt2"

# python train_8_bit.py config/train_shakespeare_8_bit_gpt2.py --compile=False --out_dir="$scratch_out_dir/out-shakespeare-8-bit-gpt2"

########################################################################################

