#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Product Image Description Inference Script
遍历所有组合：{Arts, Sports, Toys} x {train, test} x {both, img_only}
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


PRODUCT_DESCRIPTION_PROMPT = """Look at this product image and describe it in 2-4 natural sentences. Include:
- What the product is (category, brand/name if visible)
- Appearance (color, material, shape/design)
- Any visible text or labels
- Notable features or details

Requirements:
1. Only describe what is actually visible in the image
2. Keep it concise and natural, like describing the product to a friend
3. Do not guess or assume information that isn't clearly shown"""


# 遍历组合
DATASETS = ["Toys"] #"Arts", "Sports", 
SPLITS = ["train", "test"]
TYPES = ["both", "img_only"]


def load_data(data_path):
    """加载 JSON 或 JSONL 格式的数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split("\n") if line.strip()]
    return data


def load_model_with_lora(args):
    """加载基础模型并应用 LoRA checkpoint"""
    print(f"Loading base model from: {args.model_path}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="flash_attention_2" if not args.disable_flash_attn else "sdpa"
    )
    
    if args.lora_path:
        print(f"Loading LoRA checkpoint from: {args.lora_path}")
        model = PeftModel.from_pretrained(
            model,
            args.lora_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        
        if args.merge_lora:
            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()
        
        print("✅ LoRA checkpoint loaded successfully!")
    
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    
    return model, processor


def build_image_description_messages(image_path):
    """构建用于图片描述的 messages"""
    content = [
        {"type": "image", "image": image_path},
        {"type": "text", "text": PRODUCT_DESCRIPTION_PROMPT}
    ]
    return [{"role": "user", "content": content}]


def process_single_dataset(model, processor, data_path, image_folder, output_path, args):
    """处理单个数据集"""
    print(f"\n{'='*60}")
    print(f"Processing: {data_path}")
    print(f"Image folder: {image_folder}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(data_path):
        print(f"⚠️ Data file not found, skipping: {data_path}")
        return
    
    data = load_data(data_path)
    print(f"Total samples: {len(data)}")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    with open(output_path, "w", encoding="utf-8") as fout:
        for entry in tqdm(data, desc="Generating descriptions"):
            sample_id = entry.get("id", "unknown")
            
            images = entry.get("image", [])
            if isinstance(images, str):
                images = [images]
            
            image_outputs = []
            
            for img in images:
                img_path = os.path.join(image_folder, img)
                
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found: {img_path}")
                    image_outputs.append({
                        "image": img,
                        "description": "[ERROR: Image not found]"
                    })
                    continue
                
                try:
                    messages = build_image_description_messages(img_path)
                    
                    chat_text = processor.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    inputs = processor(
                        text=[chat_text],
                        images=image_inputs if image_inputs else None,
                        videos=video_inputs if video_inputs else None,
                        padding=True,
                        return_tensors="pt"
                    ).to(model.device)
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature if args.do_sample else None,
                            top_p=args.top_p if args.do_sample else None,
                        )
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] 
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    description = processor.batch_decode(
                        generated_ids_trimmed, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )[0]
                    
                    image_outputs.append({
                        "image": img,
                        "description": description
                    })
                    
                except Exception as e:
                    print(f"Error processing image {img} for sample {sample_id}: {e}")
                    image_outputs.append({
                        "image": img,
                        "description": f"[ERROR: {str(e)}]"
                    })
            
            result = {
                "id": sample_id,
                "image_descriptions": image_outputs
            }
            
            results.append(result)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
    
    print(f"✅ Completed: {len(results)} samples -> {output_path}")


def run_inference(args):
    # 加载模型（只加载一次）
    model, processor = load_model_with_lora(args)
    
    # 遍历所有组合: dataset x split x type
    for dataset in DATASETS:
        # 图片目录: ${MULTIMODAL_DATA_ROOT}/img/{dataset}_all
        image_folder = os.path.join(args.image_root, f"{dataset}_all")
        
        for split in SPLITS:
            for dtype in TYPES:
                # 输入: ${DATA_ROOT}/qwen2.5-conversations/{dataset}/{split}_all_hist_{dtype}.json
                data_path = os.path.join(
                    args.data_root,
                    dataset,
                    f"{split}_all_hist_{dtype}.json"
                )
                
                # 输出: {output_root}/{dataset}/{split}_all_hist_{dtype}_img_desc.jsonl
                output_path = os.path.join(
                    args.output_root,
                    dataset,
                    f"{split}_all_hist_{dtype}_img2txt.jsonl"
                )
                
                process_single_dataset(
                    model, processor, 
                    data_path, image_folder, output_path, 
                    args
                )
    
    print(f"\n{'='*60}")
    print("🎉 All datasets processed!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Product Image Description Inference Script")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Qwen2.5-VL base model")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA checkpoint (optional)")
    parser.add_argument("--merge_lora", action="store_true",
                        help="Merge LoRA weights into base model")
    
    # 数据参数
    parser.add_argument("--data_root", type=str, 
                        default="${DATA_ROOT}/qwen2.5-conversations",
                        help="Root path for conversation data")
    parser.add_argument("--image_root", type=str, 
                        default="${MULTIMODAL_DATA_ROOT}/img",
                        help="Root path for images")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root path for output files")
    
    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--do_sample", action="store_true",
                        help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    
    # 其他参数
    parser.add_argument("--disable_flash_attn", action="store_true",
                        help="Disable flash attention")
    
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()