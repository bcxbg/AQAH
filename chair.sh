# CUDA_VISIBLE_DEVICES=0 python final_generate_llava_chair.py --type_prompt I2 --type_decoding beam --type_method AQAH
# CUDA_VISIBLE_DEVICES=2 python final_generate_llava_chair.py --type_prompt I2 --type_decoding top_p --type_method AQAH
# CUDA_VISIBLE_DEVICES=0 python final_generate_llava_chair.py --type_prompt I1 --type_decoding beam --type_method AQAH
# CUDA_VISIBLE_DEVICES=2 python final_generate_llava_chair.py --type_prompt I1 --type_decoding top_p --type_method AQAH

CUDA_VISIBLE_DEVICES=0 python final_generate_llava_chair.py --type_prompt I2 --type_decoding beam --type_method original
CUDA_VISIBLE_DEVICES=1 python final_generate_llava_chair.py --type_prompt I2 --type_decoding top_p --type_method original
CUDA_VISIBLE_DEVICES=2 python final_generate_llava_chair.py --type_prompt I1 --type_decoding beam --type_method original
CUDA_VISIBLE_DEVICES=6 python final_generate_llava_chair.py --type_prompt I1 --type_decoding top_p --type_method original




# CUDA_VISIBLE_DEVICES=5 python final_generate_blip2_chair.py --type_prompt I2 --type_decoding beam --type_method AQAH
# CUDA_VISIBLE_DEVICES=5 python final_generate_blip2_chair.py --type_prompt I2 --type_decoding top_p --type_method AQAH
# CUDA_VISIBLE_DEVICES=5 python final_generate_blip2_chair.py --type_prompt I1 --type_decoding beam --type_method AQAH
# CUDA_VISIBLE_DEVICES=5 python final_generate_blip2_chair.py --type_prompt I1 --type_decoding top_p --type_method AQAH

# CUDA_VISIBLE_DEVICES=5 python final_generate_blip2_chair.py --type_prompt I2 --type_decoding beam --type_method original
# CUDA_VISIBLE_DEVICES=5 python final_generate_blip2_chair.py --type_prompt I2 --type_decoding top_p --type_method original
# CUDA_VISIBLE_DEVICES=5 python final_generate_blip2_chair.py --type_prompt I1 --type_decoding beam --type_method original
# CUDA_VISIBLE_DEVICES=5 python final_generate_blip2_chair.py --type_prompt I1 --type_decoding top_p --type_method original

