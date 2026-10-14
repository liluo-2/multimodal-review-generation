import torch
from torch.utils.data import Dataset
from dataloader import DescGenDataset
import json
from PIL import Image
import os
import glob

def _sanitize_text(value, default="No description provided."):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _nonempty_str(value):
    """Return stripped string or None if missing/empty."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def load_hist_img_desc_map(path):
    """
    Load parent_asin -> image description from hist_img2txt.jsonl (one JSON object per line).
    Lines without parent_asin or without non-empty description are skipped.
    """
    if not path or not str(path).strip():
        return {}
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pa = str(rec.get("parent_asin", "")).strip()
            desc = str(rec.get("description", "")).strip()
            if pa and desc:
                m[pa] = desc
    return m


def build_revgen_prompt_text(
    entry,
    sample_id,
    has_desc,
    num_hist,
    image_desc_map=None,
    img_only_ids=None,
    hist_img_desc_map=None,
):
    """
    Build the same prompt as RevGenDataset (excluding bos_token / output).
    Returns (prompt_type, prompt) where prompt_type is 'text_only' | 'both' | 'img_only'.
    """
    image_desc_map = image_desc_map or {}
    img_only_ids = set(img_only_ids or [])
    hist_img_desc_map = hist_img_desc_map or {}

    history = entry["history"]
    reference = entry["reference"]
    if num_hist != "all":
        history = history[-int(num_hist) :]

    has_image_desc_for_sample = sample_id in image_desc_map
    is_img_only = has_image_desc_for_sample and (sample_id in img_only_ids)

    if is_img_only:
        prompt_type = "img_only"
    elif has_image_desc_for_sample:
        prompt_type = "both"
    else:
        prompt_type = "text_only"

    if is_img_only:
        instruction = (
            "Given input, including the items purchased by the user historically, and corresponding reviews and ratings. "
            "Please generate a review for the new item purchased by the user, drawing from their historical reviews, ratings, and the provided image descriptions. "
            "Keep in mind that lower ratings typically result in poorer reviews."
        )
    elif has_image_desc_for_sample:
        instruction = (
            "Given input, including the items purchased by the user historically, and corresponding reviews and ratings. "
            "Please generate a review for the new item purchased by the user, drawing from their historical reviews, ratings, and the provided product and image descriptions. "
            "Keep in mind that lower ratings typically result in poorer reviews."
        )
    else:
        instruction = (
            "Given input, including the items purchased by the user historically, and corresponding reviews and ratings. "
            "Please generate a review for the new item purchased by the user, drawing from their historical reviews, ratings, and the provided product descriptions. "
            "Keep in mind that lower ratings typically result in poorer reviews."
        )

    history_lines = []
    for i, item in enumerate(history, 1):
        product_title = item["product_title"]
        text = item["text"]
        rating = item["rating"]
        parts = [f"({i}) {product_title}."]
        pd = _nonempty_str(item.get("product_desc"))
        pa = str(item.get("parent_asin") or "").strip()
        hist_img = None
        if pa:
            hist_img = _nonempty_str(hist_img_desc_map.get(pa))

        if prompt_type == "both":
            if pd:
                parts.append(f" %Product description: {pd}.")
            if hist_img:
                parts.append(f" %Image description: {hist_img}.")
        elif prompt_type == "img_only":
            if hist_img:
                parts.append(f" %Image description: {hist_img}.")
        else:
            if pd:
                parts.append(f" %Product description: {pd}.")
        parts.append(f" %Review: {text} %Rating: {rating:.1f}.")
        history_lines.append("".join(parts))
    history_block = " ".join(history_lines)

    image_desc = (
        _sanitize_text(image_desc_map.get(sample_id), default="No image description provided.")
        if has_image_desc_for_sample
        else ""
    )

    if is_img_only:
        input_block = (
            f"Item purchased by users historically: {history_block}\n\n"
            f"### The user purchased a new item:\n"
            f"%Rating: {reference['rating']:.1f}. %Image description: {image_desc}."
        )
    elif has_image_desc_for_sample:
        if has_desc:
            desc = _sanitize_text(reference.get("product_desc"))
            input_block = (
                f"Item purchased by users historically: {history_block}\n\n"
                f"### The user purchased a new item:\n"
                f"{reference['product_title']}. %Rating: {reference['rating']:.1f}. "
                f"%Product description: {desc}. %Image description: {image_desc}."
            )
        else:
            input_block = (
                f"Item purchased by users historically: {history_block}\n\n"
                f"### The user purchased a new item:\n"
                f"{reference['product_title']}. %Rating: {reference['rating']:.1f}. %Image description: {image_desc}."
            )
    else:
        desc = _sanitize_text(reference.get("product_desc"))
        input_block = (
            f"Item purchased by users historically: {history_block}\n\n"
            f"### The user purchased a new item:\n"
            f"{reference['product_title']}. %Rating: {reference['rating']:.1f}. %Product description: {desc}."
        )

    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_block}\n\n### Response:\n"
    return prompt_type, prompt


class DescDataset(Dataset):
    def __init__(self, csv_path, split_ids, tokenizer, max_length=4096):
        self.dataset = DescGenDataset(csv_path, split_ids)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        instruction = (
            "Generate a concise and attractive product description paragraph given the product's title and features. "
            "Only one continuous paragraph is allowed. Do not include any explanations, summaries, or comments. "
        )

        input_text = f"Product Title: {item['title']}\nProduct Features: {item['features']}."

        output = item["description"]  # groundtruth description

        prompt = (
            "### Instruction:\n" + instruction + "\n\n" +
            "### Input:\n" + input_text + "\n\n" +
            "### Response:\n"
        )
        # full_text = prompt + output
        full_text = prompt + output + (self.tokenizer.eos_token or "<|eot_id|>")

        # prompt = (
        #     "Given the product's title and features, generate a product description paragraph that is concise and attractive. "
        #     "Only one continuous paragraph is allowed, highlighting the main features and selling points of the product. "
        #     "Do not include any notes, summaries, explanations, or comments. Do not repeat the features or output multiple short sentences.\n\n"
        #     f"Product Title: {item['title']}\n"
        #     f"Product Features: {item['features']}.\n\n"
        #     "Product Description: "
        # )

        # description = item['description']
        # full_text = prompt + description

        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False
        )

        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        labels = input_ids.clone()

        # Prompt部分mask掉
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
        labels[:prompt_len] = -100

        # 同步截断（保证三者长度一致）
        if input_ids.size(0) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class RevGenDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        split_ids,
        tokenizer,
        max_length=4096,
        is_train=False,
        has_desc=False,
        model=None,
        num_hist="all",
        image_desc_map=None,
        img_only_ids=None,
        hist_img_desc_map=None,
        debug_print_prompts_n=0,
        better_teacher_map=None,
        is_opd=False,
    ):
        self.model = model  # kept for backwards-compat; not used with input_ids path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.split_ids = split_ids
        self.num_hist = num_hist
        self.image_desc_map = image_desc_map or {}
        self.img_only_ids = set(img_only_ids or [])
        self.hist_img_desc_map = hist_img_desc_map or {}
        self.debug_print_prompts_n = int(debug_print_prompts_n or 0)
        self.better_teacher_map = better_teacher_map or {}
        self._printed_prompts_count_by_type = {
            "text_only": 0,
            "img_only": 0,
            "both": 0,
        }
        self._printed_teacher_count = 0
        self.is_opd = is_opd

        # 读取全部数据
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line.strip()) for line in f]

        # Convert split_ids to integers if they are strings (e.g., numpy.str_)
        split_ids = [int(i) for i in split_ids]

        # 只保留属于 split 的样本（test / train / negative）
        self.data = [all_data[i] for i in split_ids]
        self.has_desc = has_desc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        sample_id = str(self.split_ids[idx])
        prompt_type, prompt = build_revgen_prompt_text(
            entry,
            sample_id,
            self.has_desc,
            self.num_hist,
            self.image_desc_map,
            self.img_only_ids,
            self.hist_img_desc_map,
        )

        # Per-sample teacher routing: "mix"/"tie" → mix teacher; "single" → modality-specific single teacher
        better = self.better_teacher_map.get(sample_id, "mix")
        if better == "single":
            teacher_key = f"{prompt_type}_single"  # "both_single", "img_only_single", "text_only_single"
        else:
            teacher_key = "mix"

        if self.debug_print_prompts_n > 0 and self._printed_teacher_count < self.debug_print_prompts_n:
            self._printed_teacher_count += 1
            print(
                f"[RevGenDataset] sample #{self._printed_teacher_count}: "
                f"id={sample_id}  modality={prompt_type}  "
                f"better_teacher={better!r}  → teacher_key={teacher_key!r}"
            )

        printed_n = self._printed_prompts_count_by_type[prompt_type]
        _hist = entry["history"]
        if self.num_hist != "all":
            _hist = _hist[-int(self.num_hist) :]
        if self.debug_print_prompts_n > 0 and printed_n < self.debug_print_prompts_n:
            self._printed_prompts_count_by_type[prompt_type] += 1
            print("\n" + "=" * 60)
            print(
                f"[RevGenDataset] Debug {prompt_type} prompt "
                f"{self._printed_prompts_count_by_type[prompt_type]}/{self.debug_print_prompts_n} "
                f"(phase={'train' if self.is_train else 'inference'}, num_hist={self.num_hist}, history_count={len(_hist)}):"
            )
            print("=" * 60)
            print(prompt)
            print("=" * 60 + "\n")

        # ── Decide what `full_text` is ─────────────────────────────────────────
        # SFT training / SFT eval: prompt + ground-truth output (for teacher forcing CE loss)
        # OPD training:            prompt only (student will generate response on-policy)
        # Inference:               prompt only
        if self.is_train and not self.is_opd:
            output = entry["reference"]["text"]
            full_text = (self.tokenizer.bos_token) + prompt + output + (self.tokenizer.eos_token or "<|eot_id|>")
        else:
            full_text = (self.tokenizer.bos_token) + prompt

        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)

        # ── SFT training / SFT eval: prompt + GT, build labels with prompt masked ─
        if self.is_train and not self.is_opd:
            labels = input_ids.clone()
            full_prompt = (self.tokenizer.bos_token) + prompt
            prompt_len = len(self.tokenizer(full_prompt, add_special_tokens=False).input_ids)
            labels[:prompt_len] = -100  # mask out prompt tokens, keep GT tokens for CE loss

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "input_type": teacher_key,
                "modality": prompt_type,
            }

        # ── OPD training: prompt only, no labels (student generates on-policy) ─
        if self.is_train and self.is_opd:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "input_type": teacher_key,
                "modality": prompt_type,
            }

        # ── Inference: prompt only, with meta_id for output tracking ───────────
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "meta_id": self.split_ids[idx],
            "input_type": teacher_key,
            "modality": prompt_type,
        }

class MMRevGenDataset(Dataset):
    def __init__(self, jsonl_path, split_ids, tokenizer, max_length=4096, is_train=False, image_root=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.split_ids = split_ids
        self.image_root = image_root

        # 读取全部数据
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line.strip()) for line in f]

        # 只保留属于 split 的样本（test / train / negative）
        self.data = [all_data[i] for i in split_ids]

    def __len__(self):
        return len(self.data)

    def _load_user_images(self, user_id, image_root):
        print("user_id:", user_id)  # 在 SFTDataset.py 的 __getitem__ 中
        image_path = os.path.join(image_root, f"{user_id}.jpg")

        # pattern = os.path.join(image_root, f"{user_id}*.jpg")
        # print("pattern:", pattern)  # 在 SFTDataset.py 的 __getitem__ 中
        # all_paths = glob.glob(pattern)

        # # 严格筛选：文件名必须以 user_id 开头（包括括号形式）
        # matched_paths = [
        #     path for path in all_paths
        #     if os.path.basename(path).startswith(user_id)
        # ][:3]

        # if not matched_paths:
        #     print(f"[Warning] No images found for user {user_id} in {image_root}")

        # images = []
        # for path in matched_paths:
        #     try:
        #         img = Image.open(path).convert("RGB")
        #         images.append(img)
        #     except Exception as e:
        #         print(f"[Warning] Failed to load image {path}: {e}")
        # while len(images) < 3:
        #     images.append(Image.new("RGB", (224, 224), color=(255, 255, 255)))

        # images = []
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                # images.append(img)
            except Exception as e:
                print(f"[Warning] Failed to load image {image_path}: {e}")
        else:
            print(f"[Warning] Image not found for user {user_id}: {image_path}")

        return img

    def __getitem__(self, idx):
        entry = self.data[idx]
        history = entry['history']
        reference = entry['reference']
        user_id = entry['user_id']
        img = self._load_user_images(user_id, self.image_root)

        # 构造 prompt
        instruction = (
            "Given input, including the items purchased by the user historically, and corresponding reviews and ratings. "
            "Please generate a review for the new item purchased by the user, drawing from their historical reviews and ratings. "
            "Keep in mind that lower ratings typically result in poorer reviews."
            "Incorporate relevant visual information from the product image when appropriate." #加了这句话
        )

        history_lines = []
        for i, item in enumerate(history, 1):
            product_title = item['product_title']
            text = item['text']
            rating = item['rating']
            history_lines.append(f"({i}) {product_title}. %Review: {text} %Rating: {rating:.1f}.")
        history_block = ' '.join(history_lines)

        # image_tokens = "<image> <image> <image>"

        image_tokens = "<image>"

        input_block = (
            f"Item purchased by users historically: {history_block}\n\n"
            f"### The user purchased a new item:\n"
            f"{reference['product_title']}. %Rating: {reference['rating']:.1f}. \nProduct images: {image_tokens}" #加了image
        )

        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_block}\n\n### Response:\n"

        # 拼接 groundtruth 输出（仅训练时）
        if self.is_train:
            output = reference['text']
            full_text = prompt + output + (self.tokenizer.eos_token or "<|eot_id|>")
        else:
            full_text = prompt

        # Tokenize（暂不加入图像，图像会在 forward 传入）
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)

        # ========== 输出 ==========
        if self.is_train:
            labels = input_ids.clone()
            prompt_len = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
            labels[:prompt_len] = -100  # mask掉 prompt

            if input_ids.size(0) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                labels = labels[:self.max_length]

            # print( {        "input_ids": input_ids,
            #                 "attention_mask": attention_mask,
            #                 "labels": labels,
            #                 "image": img  # ✅ 新增图像字段
            #             })

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "image": img  # ✅ 新增图像字段
            }
        else:
            if input_ids.size(0) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "meta_id": self.split_ids[idx],
                "image": img  # ✅ 新增图像字段
            }

class QwenRevGenDataset(Dataset):
    def __init__(self, jsonl_path, split_ids, tokenizer, max_length=8192, is_train=False, image_root=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.split_ids = split_ids
        self.image_root = image_root

        # 读取全部数据
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line.strip()) for line in f]

        # 只保留属于 split 的样本
        self.data = [all_data[i] for i in split_ids]

    def __len__(self):
        return len(self.data)

    def _load_user_images(self, user_id, image_root):
        """Load product image for the user"""
        print("user_id:", user_id)
        image_path = os.path.join(image_root, f"{user_id}.jpg")
        
        if not os.path.exists(image_path):
            print(f"[Warning] Image not found for user {user_id}: {image_path}")
        #     try:
        #         img = Image.open(image_path).convert("RGB")
        #         return img
        #     except Exception as e:
        #         print(f"[Warning] Failed to load image {image_path}: {e}")
        #         # Return a white placeholder image if loading fails
        #         return Image.new("RGB", (224, 224), color=(255, 255, 255))
        # else:
        #     print(f"[Warning] Image not found for user {user_id}: {image_path}")
        #     # Return a white placeholder image if not found
        #     return Image.new("RGB", (224, 224), color=(255, 255, 255))

    def _format_qwen_vl_conversation(self, instruction, history_block, reference, image_path=None):
        """Format conversation for Qwen-VL with proper image tokens"""
        
        # Build the system instruction
        system_msg = (
            "Given input, including the items purchased by the user historically, and corresponding reviews and ratings. "
            "Please generate a review for the new item purchased by the user, drawing from their historical reviews and ratings. "
            "Keep in mind that lower ratings typically result in poorer reviews. "
            "Incorporate relevant visual information from the product image when appropriate."
        )
        
        # Build user input with image
        user_input = (
            f"Item purchased by users historically: {history_block}\n\n"
            f"### The user purchased a new item:\n"
            f"{reference['product_title']}. %Rating: {reference['rating']:.1f}.\n"
        )
        
        # Add image token for Qwen-VL
        if image_path:
            user_input += f"Product images: <img>{image_path}</img>\n"
        else:
            user_input += "Product images: <img>placeholder</img>\n"
        
        user_input += "\n### Response: "
        
        # Format as Qwen-VL conversation
        # Qwen-VL format: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>
        conversation = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
        return conversation

    def __getitem__(self, idx):
        entry = self.data[idx]
        history = entry['history']
        reference = entry['reference']
        user_id = entry['user_id']
        
        # Load image
        img = self._load_user_images(user_id, self.image_root)
        image_path = os.path.join(self.image_root, f"{user_id}.jpg") if self.image_root else "placeholder"

        # Build history block
        history_lines = []
        for i, item in enumerate(history, 1):
            product_title = item['product_title']
            text = item['text']
            rating = item['rating']
            history_lines.append(f"({i}) {product_title}. %Review: {text} %Rating: {rating:.1f}.")
        history_block = ' '.join(history_lines)

        # Format conversation for Qwen-VL
        conversation = self._format_qwen_vl_conversation(
            instruction="", 
            history_block=history_block, 
            reference=reference,
            image_path=image_path
        )

        if self.is_train:
            # Add assistant response and end token
            output = reference['text']
            full_conversation = conversation + output + "<|im_end|>"
            
            # Tokenize full conversation
            tokenized = self.tokenizer(
                full_conversation,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
                add_special_tokens=False
            )
            
            input_ids = tokenized.input_ids.squeeze(0)
            attention_mask = tokenized.attention_mask.squeeze(0)
            
            # Create labels - mask everything except the assistant response
            labels = input_ids.clone()
            
            # Find where assistant response starts
            assistant_start_text = "<|im_start|>assistant\n"
            assistant_tokens = self.tokenizer(assistant_start_text, add_special_tokens=False).input_ids
            
            # Mask prompt tokens (everything before assistant response)
            prompt_conversation = conversation
            prompt_tokens = self.tokenizer(prompt_conversation, add_special_tokens=False).input_ids
            prompt_len = len(prompt_tokens)
            
            if prompt_len < len(labels):
                labels[:prompt_len] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
                # "images": img,
                # "image_sizes": [img.size],
            }
        else:
            # For inference, only tokenize the conversation up to assistant
            tokenized = self.tokenizer(
                conversation,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
                add_special_tokens=False
            )
            
            input_ids = tokenized.input_ids.squeeze(0)
            attention_mask = tokenized.attention_mask.squeeze(0)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                # "images": img,
                # "image_sizes": [img.size],
                "meta_id": self.split_ids[idx]
            }

class MMRevGenPrecomputedDataset(Dataset):
    def __init__(self, prompt_path, tokenizer, emb_root, max_length, is_train=True):
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load json and precomputed embeddings
        self.prompts = [json.loads(line) for line in open(prompt_path)]

        self.text_embs = torch.load(f"{emb_root}/text_embeddings.pt")
        self.img_embs = torch.load(f"{emb_root}/image_embeddings.pt")
        self.hist_text_embs = torch.load(f"{emb_root}/hist_text_embeddings.pt")
        self.hist_img_embs = torch.load(f"{emb_root}/hist_image_embeddings.pt")

    def __len__(self): return len(self.prompts)

    def __getitem__(self, idx):
        rec = self.prompts[idx]
        rid = rec["id"]
        text_emb = self.text_embs[rid]
        img_emb = self.img_embs[rid]
        hist_text_emb = self.hist_text_embs.get(rid)
        hist_img_emb = self.hist_img_embs.get(rid)

        prompt = rec["prompt"]
        review = rec.get("gt", "")  # 有时推理阶段可能无 gt
        input_text = self.tokenizer.bos_token + prompt

        if self.is_train:
            input_text += review + self.tokenizer.eos_token

        toks = self.tokenizer(
            input_text,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length
        )
        input_ids = toks.input_ids.squeeze(0)
        attn_mask = toks.attention_mask.squeeze(0)

        # 仅在训练时构造 labels
        if self.is_train:
            labels = input_ids.clone()
            prompt_ids = self.tokenizer(
                        self.tokenizer.bos_token + prompt,
                        add_special_tokens=False
                    ).input_ids
            # prompt_len = len(self.tokenizer(prompt).input_ids)
            prompt_len = len(prompt_ids)
            labels[:prompt_len] = -100
        else:
            labels = None

        sample = {
            "id": rid,
            "gt": review,
            "text_emb": text_emb,
            "img_emb": img_emb,
            "hist_text_emb": hist_text_emb,
            "hist_img_emb": hist_img_emb,
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        }

        if labels is not None:
            sample["labels"] = labels

        return sample