CUDA_VISIBLE_DEVICES=4 python final_generate_pope.py --type_question adversarial --type_dataset aokvqa --type_method original
CUDA_VISIBLE_DEVICES=5 python final_generate_pope.py --type_question random --type_dataset aokvqa --type_method original
CUDA_VISIBLE_DEVICES=6 python final_generate_pope.py --type_question popular --type_dataset aokvqa --type_method original


# CUDA_VISIBLE_DEVICES=5 python final_generate_pope.py --type_question popular --type_dataset gqa --type_method original
# CUDA_VISIBLE_DEVICES=5 python final_generate_pope.py --type_question adversarial --type_dataset gqa --type_method original
# CUDA_VISIBLE_DEVICES=5 python final_generate_pope.py --type_question random --type_dataset gqa --type_method original


# CUDA_VISIBLE_DEVICES=4 python final_generate_pope.py --type_question adversarial --type_dataset coco --type_method original
# CUDA_VISIBLE_DEVICES=4 python final_generate_pope.py --type_question random --type_dataset coco --type_method original
# CUDA_VISIBLE_DEVICES=4 python final_generate_pope.py --type_question popular --type_dataset coco --type_method original


CUDA_VISIBLE_DEVICES=6 python final_generate_pope.py --type_question adversarial --type_dataset aokvqa --type_method AQAH
CUDA_VISIBLE_DEVICES=4 python final_generate_pope.py --type_question random --type_dataset aokvqa --type_method AQAH
CUDA_VISIBLE_DEVICES=5 python final_generate_pope.py --type_question popular --type_dataset aokvqa --type_method AQAH


CUDA_VISIBLE_DEVICES=4 python final_generate_pope.py --type_question popular --type_dataset gqa --type_method AQAH
CUDA_VISIBLE_DEVICES=5 python final_generate_pope.py --type_question adversarial --type_dataset gqa --type_method AQAH
CUDA_VISIBLE_DEVICES=5 python final_generate_pope.py --type_question random --type_dataset gqa --type_method AQAH


CUDA_VISIBLE_DEVICES=4 python final_generate_pope.py --type_question adversarial --type_dataset coco --type_method AQAH
CUDA_VISIBLE_DEVICES=2 python final_generate_pope.py --type_question random --type_dataset coco --type_method AQAH
CUDA_VISIBLE_DEVICES=4 python final_generate_pope.py --type_question popular --type_dataset coco --type_method AQAH


