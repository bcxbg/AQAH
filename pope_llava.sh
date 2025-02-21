CUDA_VISIBLE_DEVICES=7 python final_generate_pope_llava.py --type_question adversarial --type_dataset gqa --type_method original
CUDA_VISIBLE_DEVICES=0 python final_generate_pope_llava.py --type_question random --type_dataset gqa --type_method original
CUDA_VISIBLE_DEVICES=0 python final_generate_pope_llava.py --type_question popular --type_dataset gqa --type_method original

CUDA_VISIBLE_DEVICES=0 python final_generate_pope_llava.py --type_question adversarial --type_dataset aokvqa --type_method original
CUDA_VISIBLE_DEVICES=0 python final_generate_pope_llava.py --type_question random --type_dataset aokvqa --type_method original
CUDA_VISIBLE_DEVICES=7 python final_generate_pope_llava.py --type_question popular --type_dataset coco --type_method original





CUDA_VISIBLE_DEVICES=0 python final_generate_pope_llava.py --type_question popular --type_dataset aokvqa --type_method original
CUDA_VISIBLE_DEVICES=7 python final_generate_pope_llava.py --type_question adversarial --type_dataset coco --type_method original
CUDA_VISIBLE_DEVICES=0 python final_generate_pope_llava.py --type_question random --type_dataset coco --type_method original
