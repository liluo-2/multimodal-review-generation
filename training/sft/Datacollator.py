from typing import Any, List, Dict
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Any, List, Dict
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Any, List, Dict
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence

def _transpose_hist(batch_list):  # [B][H] -> [H][B]，元素可为 Tensor 或 None
    B = len(batch_list)
    maxH = max(len(x) if x is not None else 0 for x in batch_list)
    slots = []
    for s in range(maxH):
        slot_s = []
        for b in range(B):
            xb = batch_list[b]
            tb = xb[s] if (xb is not None and s < len(xb) and xb[s] is not None) else None
            slot_s.append(tb)
        slots.append(slot_s)
    return slots  # [H][B]

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ── Determine the scenario ────────────────────────────────────────────
        has_labels = "labels" in features[0]
        has_meta_id = "meta_id" in features[0]
        # Pure inference = no labels + has meta_id  → needs LEFT padding (for generate)
        # All other cases (SFT train/eval, OPD train) → RIGHT padding
        is_inference = (not has_labels) and has_meta_id

        # ── input_ids path ────────────────────────────────────────────────────
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        # Switch tokenizer padding side based on scenario
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left" if is_inference else "right"
        try:
            enc = self.tokenizer.pad(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                padding="longest",
                pad_to_multiple_of=self.pad_to_multiple_of if not is_inference else None,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        batch = {k: v for k, v in enc.items()}

        if has_meta_id:
            batch["meta_id"] = [f.get("meta_id") for f in features]
        if "input_type" in features[0]:
            batch["input_type"] = [f["input_type"] for f in features]

        if has_labels:
            labels_list = [f["labels"] for f in features]
            labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
            T_main = batch["input_ids"].size(1)
            if labels.size(1) < T_main:
                pad = torch.full(
                    (labels.size(0), T_main - labels.size(1)),
                    -100, dtype=labels.dtype, device=labels.device,
                )
                labels = torch.cat([labels, pad], dim=1)
            else:
                labels = labels[:, :T_main]
            batch["labels"] = labels

        return batch

# @dataclass
# class DataCollatorForSupervisedDataset:
#     tokenizer: any
#     pad_to_multiple_of: int = 8  # Optional: TensorCore加速用，不强制

#     def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#         # 单独提取各字段
#         input_ids = [f["input_ids"] for f in features]
#         attention_mask = [f["attention_mask"] for f in features]
#         labels = [f["labels"] for f in features]

#         # pad input_ids 和 attention_mask，直接用 tokenizer.pad（标准处理input）
#         batch = self.tokenizer.pad(
#             {"input_ids": input_ids, "attention_mask": attention_mask},
#             padding="longest",
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt"
#         )

#         # pad labels，自己用 torch.pad_sequence
#         labels = pad_sequence(
#             labels,
#             batch_first=True,
#             padding_value=-100  # Mask label的padding部分
#         )

#         if self.pad_to_multiple_of is not None:
#             padding_size = labels.size(1)
#             if padding_size % self.pad_to_multiple_of != 0:
#                 pad_len = self.pad_to_multiple_of - (padding_size % self.pad_to_multiple_of)
#                 # 手动在后面补-100
#                 pad_tensor = torch.full(
#                     (labels.size(0), pad_len),
#                     fill_value=-100,
#                     dtype=labels.dtype,
#                     device=labels.device
#                 )
#                 labels = torch.cat([labels, pad_tensor], dim=1)

#         # 返回标准batch
#         batch["labels"] = labels
#         return batch

# @dataclass
# class MMDataCollatorForSupervisedDataset: # multimodal collator
#     tokenizer: any
#     pad_to_multiple_of: int = 8

#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#         print("tokenizer.eos_token:", self.tokenizer.eos_token)
#         print("tokenizer.eos_token_id:", self.tokenizer.eos_token_id)
#         # print("tokenizer.additional_special_tokens:", self.tokenizer.additional_special_tokens)
#         # print("tokenizer.convert_tokens_to_ids('<image>'):", self.tokenizer.convert_tokens_to_ids("<image>"))

#         # print("features:", features)  # Debugging line
#         input_ids = [f["input_ids"] for f in features]
#         attention_mask = [f["attention_mask"] for f in features]
#         labels = [f["labels"] for f in features]
#         images = [f["image"] for f in features]  # ✅ 图像字段：PIL.Image 或 Tensor

#         if self.tokenizer.pad_token_id is None:
#             raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

#         # 使用 tokenizer.pad 对 input_ids 和 attention_mask 进行填充
#         batch = self.tokenizer.pad(
#             {"input_ids": input_ids, "attention_mask": attention_mask},
#             padding="longest",
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt"
#         )

#         # 对 labels 进行 padding（不能用 tokenizer.pad）
#         labels = pad_sequence(labels, batch_first=True, padding_value=-100)

#         # 处理 labels 长度 pad_to_multiple_of
#         if self.pad_to_multiple_of is not None:
#             seq_len = labels.size(1)
#             if seq_len % self.pad_to_multiple_of != 0:
#                 pad_len = self.pad_to_multiple_of - (seq_len % self.pad_to_multiple_of)
#                 pad_tensor = torch.full((labels.size(0), pad_len), fill_value=-100, dtype=labels.dtype)
#                 labels = torch.cat([labels, pad_tensor], dim=1)

#         batch["labels"] = labels
#         batch["images"] = images  # ✅ 多模态图像列表，后续模型支持 image=batch["images"]
#         return batch

@dataclass
class InferenceCollator:
    tokenizer: any

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        meta_ids = [f["meta_id"] for f in features]  # 保留原始ID

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            return_tensors="pt"
        )
        batch["meta_id"] = meta_ids
        return batch

@dataclass
class MMInferenceCollator:
    tokenizer: any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        meta_ids = [f["meta_id"] for f in features]
        images = [f["image"] for f in features]  # ✅ 新增图像字段

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            return_tensors="pt"
        )

        batch["meta_id"] = meta_ids
        batch["images"] = images  # ✅ 图像列表，模型自动识别
        return batch

# class QwenVLCollator:
#     def __init__(self, tokenizer, pad_to_multiple_of=8):
#         self.tokenizer = tokenizer
#         self.pad_to_multiple_of = pad_to_multiple_of
    
#     def __call__(self, batch):
#         # Extract images
#         images = [item.pop('images') for item in batch]
#         image_sizes = [item.pop('image_sizes', [None]) for item in batch]
        
#         # Pad sequences
#         max_length = max(len(item['input_ids']) for item in batch)
#         if self.pad_to_multiple_of:
#             max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
#         padded_batch = {
#             'input_ids': [],
#             'attention_mask': [],
#             'images': images,
#             'image_sizes': image_sizes
#         }
        
#         if 'labels' in batch[0]:
#             padded_batch['labels'] = []
        
#         if 'meta_id' in batch[0]:
#             padded_batch['meta_id'] = [item['meta_id'] for item in batch]
        
#         for item in batch:
#             seq_len = len(item['input_ids'])
#             pad_length = max_length - seq_len
            
#             # Pad input_ids
#             padded_input_ids = torch.cat([
#                 item['input_ids'], 
#                 torch.full((pad_length,), self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
#             ])
#             padded_batch['input_ids'].append(padded_input_ids)
            
#             # Pad attention_mask
#             padded_attention_mask = torch.cat([
#                 item['attention_mask'],
#                 torch.zeros(pad_length, dtype=torch.long)
#             ])
#             padded_batch['attention_mask'].append(padded_attention_mask)
            
#             # Pad labels if present
#             if 'labels' in item:
#                 padded_labels = torch.cat([
#                     item['labels'],
#                     torch.full((pad_length,), -100)
#                 ])
#                 padded_batch['labels'].append(padded_labels)
        
#         # Stack tensors
#         padded_batch['input_ids'] = torch.stack(padded_batch['input_ids'])
#         padded_batch['attention_mask'] = torch.stack(padded_batch['attention_mask'])
        
#         if 'labels' in padded_batch:
#             padded_batch['labels'] = torch.stack(padded_batch['labels'])
        
#         return padded_batch