import torch
import random
import numpy as np
from types import MethodType
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from peft import PeftModel, PeftConfig
import torch.nn as nn
from mm_fusion import CrossAttentionFusion, MultiHeadGMU, CrossAttentionFusionWithGate


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def load_model_and_tokenizer(model_name, model_path, tokenizer_path, is_train=True, use_lora=False, use_expert_mlp=False, expert_bottleneck_dim=1024, use_multi_head=False):
    # ======== Load Tokenizer ========
    if model_name == "llama-3":
        # 这里的 tokenizer_path 最好是你训练后的 checkpoint 目录，或者你需要确保 special tokens 添加逻辑与训练时一致
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        # ===== 添加 special tokens =====
        # 必须确保这里添加的 tokens 和训练时完全一致，这样 len(tokenizer) 才会变成 128263
        special_tokens = ["<NEWITEM>", "<HIST1>", "<HIST2>", "<HIST3>", "<HIST4>", "<HIST5>"]
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
    elif model_name == "qwen-vl":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # ======== Padding Direction ========
    tokenizer.padding_side = "right" if is_train else "left"

    # ======== Ensure Safe pad_token ========
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        print("[INFO] Adding separate <pad> token to tokenizer.")
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    print(f"[INFO] pad_token = {tokenizer.pad_token}, id = {tokenizer.pad_token_id}")
    print(f"[INFO] eos_token = {tokenizer.eos_token}, id = {tokenizer.eos_token_id}")
    print(f"[INFO] Tokenizer vocab size: {len(tokenizer)}")

    # ======== Load Model ========
    if is_train:
        # -------- Training Mode --------
        if model_name == "llama-3":
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                device_map="auto"
            )
        elif model_name == "qwen-vl":
            tokenizer.eos_token = "<|endoftext|>"
            tokenizer.pad_token = "<|endoftext|>"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True
            )

        # ⚠️ 训练阶段：resize
        model.resize_token_embeddings(len(tokenizer))
        print(f"[INFO] Model embeddings resized for training. Vocab size = {len(tokenizer)}")

    else:
        # -------- Inference Mode --------
        set_seed(42)

        if use_lora:
            # ===== LoRA 模式 =====
            # model_path 在这里指的是 LoRA adapter 的路径
            config = PeftConfig.from_pretrained(model_path)

            if model_name == "llama-3":
                base_model = LlamaForCausalLM.from_pretrained(
                    config.base_model_name_or_path, # 从 config 自动读取底座路径
                    device_map="auto",
                    torch_dtype="auto"
                )
            elif model_name == "qwen-vl":
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True
                )

            print(f"[INFO] Base model loaded: {config.base_model_name_or_path}")

            # ========================================================
            # 关键修复：在加载 Adapter 之前，必须先 Resize Base Model
            # ========================================================
            current_model_vocab = base_model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) > current_model_vocab:
                print(f"[INFO] Resizing base model embeddings from {current_model_vocab} to {len(tokenizer)} BEFORE loading LoRA.")
                base_model.resize_token_embeddings(len(tokenizer))

            # 加载 Adapter
            print(f"[INFO] Loading LoRA adapter from {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)
            
            # 合并权重 (可选，看你需要不需要)
            model = model.merge_and_unload()
            print("[INFO] LoRA adapter merged for inference.")
            
        else:
            # ===== Base 模式 (全量模型) =====
            # 如果是全量微调后的模型，直接加载即可
            if model_name == "llama-3":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype="auto"
                )
            elif model_name == "qwen-vl":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True
                )
            print(f"[INFO] Full model loaded from {model_path}")

        # ===== 双重检查 =====
        # 防止某些特殊情况没 Resize 到位
        model_vocab = model.get_input_embeddings().weight.shape[0]
        tokenizer_vocab = len(tokenizer)
        if tokenizer_vocab != model_vocab:
            print(f"[WARN] Vocabulary size mismatch after load: model={model_vocab}, tokenizer={tokenizer_vocab}")
            print("[INFO] Resizing model embeddings to match tokenizer...")
            model.resize_token_embeddings(tokenizer_vocab)
        
        # ===== 设置半精度与评估模式 =====
        # 注意：merge_and_unload 后模型已经是 bf16/fp16 了，再次 .half() 可能会导致精度损失或报错，视情况而定
        # 这里保留你的逻辑，但建议加上 hasattr 检查
        if model.dtype == torch.float32:
             model = model.half()
             
        model.eval()
        print("[INFO] Model set to evaluation mode.")

    return model, tokenizer
# def load_model_and_tokenizer(model_name, model_path, tokenizer_path, is_train=True, use_lora=False, use_expert_mlp=False, expert_bottleneck_dim=1024, use_multi_head=False):
#     if model_name == "llama-3":
#         tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
#     elif model_name == "qwen-vl":
#         tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)

#     tokenizer.padding_side = "right" if is_train else "left"
#     print("pad_token:", tokenizer.pad_token)
#     print("eos_token:", tokenizer.eos_token)
#     print("pad_token_id:", tokenizer.pad_token_id)
#     print("eos_token_id:", tokenizer.eos_token_id)

#     if tokenizer.pad_token is None:
#         # tokenizer.pad_token = tokenizer.eos_token  # fallback pad token
#         tokenizer.pad_token = "<pad>"
#         tokenizer.add_special_tokens({"pad_token": "<pad>"})


#     if is_train:
#         # ============ Training Mode ============
#         if model_name == "llama-3":
#             model = LlamaForCausalLM.from_pretrained(
#                 model_path,
#                 device_map="auto"
#             )
#             model.resize_token_embeddings(len(self.tokenizer))
#         elif model_name == "qwen-vl":
#             tokenizer.eos_token = "<|endoftext|>"
#             tokenizer.pad_token = "<|endoftext|>"
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 device_map="auto",
#                 trust_remote_code=True
#             )
#             model.resize_token_embeddings(len(tokenizer))

#     else:
#         # ============ Inference Mode ============
#         set_seed(42)

#         if use_lora:
#             # ----- Load base model first -----
#             config = PeftConfig.from_pretrained(model_path)
#             if model_name == "llama-3":
#                 base_model = LlamaForCausalLM.from_pretrained(
#                     config.base_model_name_or_path,
#                     device_map="auto",
#                     torch_dtype="auto"
#                 )
#             elif model_name == "qwen-vl":
#                 base_model = AutoModelForCausalLM.from_pretrained(
#                     config.base_model_name_or_path,
#                     device_map="auto",
#                     torch_dtype="auto",
#                     trust_remote_code=True
#                 )
#             print(f"[INFO] Base model inferred from LoRA config: {config.base_model_name_or_path}")

#             # ----- Load and merge LoRA -----
#             model = PeftModel.from_pretrained(base_model, model_path)
#             print(f"[INFO] LoRA adapter loaded from {model_path}")
#             model = model.merge_and_unload()
#             print("[INFO] LoRA adapter merged for inference.")

#         else:
#             # ----- Load standard pretrained model -----
#             if model_name == "llama-3":
#                 model = AutoModelForCausalLM.from_pretrained(
#                     model_path,
#                     device_map="auto",
#                     torch_dtype="auto"
#                 )
#             elif model_name == "qwen-vl":
#                 model = AutoModelForCausalLM.from_pretrained(
#                     model_path,
#                     device_map="auto",
#                     torch_dtype="auto",
#                     trust_remote_code=True
#                 )
#             print(f"[INFO] Full model loaded from {model_path}")

#         model = model.half()
#         model.eval()
#         print("[INFO] Model set to evaluation mode.")

#     return model, tokenizer

class MultiModalLlama3WithGate(nn.Module):
    """
    - 外部 text/img 序列在 256 维融合；CLS 进 GMU，full 序列进 Cross-Attn
    - 融合后用 projector 投到 LLaMA 的 embedding 维度
    """
    def __init__(
        self,
        llm,
        tokenizer,
        use_softmax_gating=False,
        is_inference=False,
        fusion_dim=256,
        use_cross_attn=True,
    ):  
        super().__init__()
        self.tokenizer = tokenizer
        self.llm = llm
        self.use_softmax_gating = use_softmax_gating
        self.fusion_dim = fusion_dim
        self.use_cross_attn = use_cross_attn

        # 仅读取一次维度，不缓存模块以避免共享权重保存冲突
        emb = self.llm.get_input_embeddings()
        self.llm_hidden_dim = emb.embedding_dim
        target_device = emb.weight.device
        target_dtype  = emb.weight.dtype

        # 融合模块（始终在 fusion_dim）
        if self.use_cross_attn:
            self.cross_attn = CrossAttentionFusionWithGate(fusion_dim)
        else:
            self.cross_attn = None
        self.gmu          = MultiHeadGMU(fusion_dim, num_heads=4, use_softmax=use_softmax_gating)
        self.proj_to_llm  = nn.Linear(fusion_dim, self.llm_hidden_dim)

        # learnable fallback（保证和底模一致的 dtype/device）
        self.text_fallback = nn.Parameter(
            torch.randn(1, 1, fusion_dim, device=target_device, dtype=target_dtype)
        )
        self.img_fallback  = nn.Parameter(
            torch.randn(1, 1, fusion_dim, device=target_device, dtype=target_dtype)
        )

        # special tokens
        special_tokens = ["<NEWITEM>", "<HIST1>", "<HIST2>", "<HIST3>", "<HIST4>", "<HIST5>"]
        self.placeholder_token_ids = {t: self.tokenizer.convert_tokens_to_ids(t) for t in special_tokens}
        self.prompt_expand_len = 4
        self.is_inference = is_inference

        # ✅ 把自建层整体对齐 dtype/device（包含 fusion 模块 & projector）
        self.to(device=target_device, dtype=target_dtype)

    def freeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            try:
                self.llm.gradient_checkpointing_disable()
            except Exception:
                pass

    @staticmethod
    def _cast_ext(x, device, dtype):
        """把外部 embedding（Tensor 或 list[Tensor] 或 None）统一到 device/dtype。"""
        if x is None: 
            return None
        if isinstance(x, list):
            return [MultiModalLlama3._cast_ext(xx, device, dtype) for xx in x]
        return x.to(device=device, dtype=dtype)

    def fuse_item(self, text_emb, img_emb):
        """
        输入: (B,L,256) or list[...] or None
        策略: text/img 用 CLS；cross-attn 用 full 序列；输出 (B,4,hidden_dim)
        """
        device = self.text_fallback.device
        dtype  = self.text_fallback.dtype

        # 如果传入 None，在下面会 fallback；若非 None，先 cast dtype/device
        text_emb = self._cast_ext(text_emb, device, dtype)
        img_emb  = self._cast_ext(img_emb,  device, dtype)

        # 统一 batch 维度
        if isinstance(text_emb, list) or isinstance(img_emb, list):
            B = len(text_emb) if isinstance(text_emb, list) else len(img_emb)
            t_list, v_list = [], []
            for t, v in zip(text_emb if isinstance(text_emb, list) else [None]*B,
                            img_emb  if isinstance(img_emb,  list) else [None]*B):
                t_list.append(t if t is not None else self.text_fallback)
                v_list.append(v if v is not None else self.img_fallback)
            text_emb = torch.cat(t_list, dim=0)
            img_emb  = torch.cat(v_list, dim=0)
        else:
            # 单 Tensor 情况
            B = (text_emb.size(0) if text_emb is not None else (img_emb.size(0) if img_emb is not None else 1))
            if text_emb is None:
                text_emb = self.text_fallback.expand(B, 1, -1)
            if img_emb is None:
                img_emb  = self.img_fallback.expand(B, 1, -1)

        # 拆 CLS 与 full 序列
        text_cls = text_emb[:, 0:1, :]   # (B,1,256)
        img_cls  = img_emb[:, 0:1, :]    # (B,1,256)
        text_seq = text_emb               # (B,Lt,256)
        img_seq  = img_emb                # (B,Li,256)

        # Cross-Attn + GMU（都已对齐 dtype/device）
        if self.use_cross_attn and self.cross_attn is not None:
            cross_emb = self.cross_attn(text_seq, img_seq)                    # (B,1,256)
            fused = self.gmu(text_cls, cross_emb, img_cls)                    # (B,4,256)
        else:
            print("No cross attention is used!")
            # 如果不使用 cross attention，GMU 只接收 text_cls 和 img_cls
            fused = self.gmu(text_cls, cross_emb=None, img_emb=img_cls)       # (B,4,256)

        # 投回 LLM embedding 维度
        return self.proj_to_llm(fused)                                    # (B,4,4096)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        text_emb=None,
        img_emb=None,
        hist_text_emb=None,
        hist_img_emb=None,
        **kwargs
    ):
        attn_mask = attention_mask
        input_embeds = self.llm.get_input_embeddings()(input_ids)

        # 新物品 + 历史 5 槽
        newitem_emb = self.fuse_item(text_emb, img_emb)  # (B,4,D)
        hist_embs = []
        for ht, hi in zip(hist_text_emb, hist_img_emb):
            hist_embs.append(self.fuse_item(ht, hi))     # list of (B,4,D)

        B, L, D = input_embeds.shape
        new_input_embeds, new_attn, new_labels = [], [], []
        pid = self.placeholder_token_ids
        for b in range(B):
            seq_emb, seq_mask, seq_label = [], [], []
            for i in range(L):
                tok_id = input_ids[b, i].item()
                if tok_id == pid["<NEWITEM>"]:
                    seq_emb.append(newitem_emb[b])
                    seq_mask.append(torch.ones(self.prompt_expand_len, device=attn_mask.device, dtype=attn_mask.dtype))
                    if labels is not None:
                        seq_label.append(torch.full((self.prompt_expand_len,), -100, device=attn_mask.device, dtype=torch.long))
                    continue

                replaced = False
                for name, tid in pid.items():
                    if name.startswith("<HIST") and tok_id == tid:
                        idx = int(name[5]) - 1
                        if idx < len(hist_embs):
                            seq_emb.append(hist_embs[idx][b])
                            seq_mask.append(torch.ones(self.prompt_expand_len, device=attn_mask.device, dtype=attn_mask.dtype))
                            if labels is not None:
                                seq_label.append(torch.full((self.prompt_expand_len,), -100, device=attn_mask.device, dtype=torch.long))
                        replaced = True
                        break

                if not replaced:
                    seq_emb.append(input_embeds[b, i:i+1, :])
                    seq_mask.append(attn_mask[b, i:i+1])
                    if labels is not None:
                        seq_label.append(labels[b, i:i+1])

            seq_emb  = torch.cat(seq_emb,  dim=0)
            seq_mask = torch.cat(seq_mask, dim=0)
            seq_lab  = torch.cat(seq_label, dim=0) if labels is not None else None

            new_input_embeds.append(seq_emb)
            new_attn.append(seq_mask)
            new_labels.append(seq_lab)

        max_len = max(e.size(0) for e in new_input_embeds)
        padded_embeds, padded_masks, padded_labels = [], [], []
        for e, m, l in zip(new_input_embeds, new_attn, new_labels):
            pad_len = max_len - e.size(0)
            padded_embeds.append(torch.cat([e, torch.zeros(pad_len, D, device=e.device, dtype=e.dtype)], dim=0))
            padded_masks.append(torch.cat([m, torch.zeros(pad_len, device=m.device, dtype=m.dtype)], dim=0))
            if labels is not None:
                padded_labels.append(torch.cat([l, torch.full((pad_len,), -100, device=m.device, dtype=torch.long)], dim=0))

        input_embeds = torch.stack(padded_embeds, dim=0)
        attn_mask    = torch.stack(padded_masks, dim=0)
        labels       = torch.stack(padded_labels, dim=0) if labels is not None else None

        if labels is not None:
            return self.llm(inputs_embeds=input_embeds, attention_mask=attn_mask, labels=labels)
        else:
            return self.llm(inputs_embeds=input_embeds, attention_mask=attn_mask)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.llm.prepare_inputs_for_generation(*args, **kwargs)


class MultiModalLlama3(nn.Module):
    """
    - 外部 text/img 序列在 256 维融合；CLS 进 GMU，full 序列进 Cross-Attn
    - 融合后用 projector 投到 LLaMA 的 embedding 维度
    """
    def __init__(
        self,
        llm,
        tokenizer,
        is_inference=False,
        num_queries=32,
        fusion_dim=256,
    ):  
        super().__init__()
        self.tokenizer = tokenizer
        self.llm = llm
        self.fusion_dim = fusion_dim

        # 仅读取一次维度，不缓存模块以避免共享权重保存冲突
        emb = self.llm.get_input_embeddings()
        self.llm_hidden_dim = emb.embedding_dim
        target_device = emb.weight.device
        target_dtype  = emb.weight.dtype

        # ====== 融合模块：Q-Former 风格 Cross-Attn ======
        self.num_queries = num_queries # 每个占位符展开成 32 个 soft tokens
        self.cross_attn   = CrossAttentionFusion(fusion_dim, num_queries=self.num_queries)
        self.proj_to_llm  = nn.Linear(fusion_dim, self.llm_hidden_dim)

        # learnable fallback（保证和底模一致的 dtype/device）
        self.text_fallback = nn.Parameter(
            torch.randn(1, 1, fusion_dim, device=target_device, dtype=target_dtype)
        )
        self.img_fallback  = nn.Parameter(
            torch.randn(1, 1, fusion_dim, device=target_device, dtype=target_dtype)
        )

        # special tokens
        special_tokens = ["<NEWITEM>", "<HIST1>", "<HIST2>", "<HIST3>", "<HIST4>", "<HIST5>"]
        self.placeholder_token_ids = {t: self.tokenizer.convert_tokens_to_ids(t) for t in special_tokens}
        self.prompt_expand_len = self.num_queries   # ⭐ 每个占位符展开 32 个 token
        self.is_inference = is_inference

        # ✅ 把自建层整体对齐 dtype/device（包含 fusion 模块 & projector）
        self.to(device=target_device, dtype=target_dtype)

    def freeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            try:
                self.llm.gradient_checkpointing_disable()
            except Exception:
                pass

    @staticmethod
    def _cast_ext(x, device, dtype):
        """把外部 embedding（Tensor 或 list[Tensor] 或 None）统一到 device/dtype。"""
        if x is None: 
            return None
        if isinstance(x, list):
            return [MultiModalLlama3._cast_ext(xx, device, dtype) for xx in x]
        return x.to(device=device, dtype=dtype)

    def fuse_item(self, text_emb, img_emb):
        """
        输入:
            text_emb: (B, Lt, 256) 或 list[Tensor] 或 None
            img_emb:  (B, Li, 256) 或 list[Tensor] 或 None
        逻辑:
            1. 把 None 替换成 learnable fallback
            2. 把 list 整理成 batch Tensor
            3. 直接用 full sequence 做 cross-attn
        输出:
            (B, num_queries, llm_hidden_dim)  —— 用作 soft tokens
        """
        device = self.text_fallback.device
        dtype  = self.text_fallback.dtype

        # 统一 dtype/device
        text_emb = self._cast_ext(text_emb, device, dtype)
        img_emb  = self._cast_ext(img_emb,  device, dtype)

        # ---------- 把 list / None 整理成 (B, L, D) ----------
        if isinstance(text_emb, list) or isinstance(img_emb, list):
            B = len(text_emb) if isinstance(text_emb, list) else len(img_emb)
            t_list, v_list = [], []
            for t, v in zip(
                text_emb if isinstance(text_emb, list) else [None] * B,
                img_emb  if isinstance(img_emb,  list) else [None] * B
            ):
                if t is None:
                    t = self.text_fallback      # (1,1,D)
                if v is None:
                    v = self.img_fallback       # (1,1,D)
                if t.dim() == 2:
                    t = t.unsqueeze(0)          # -> (1,Lt,D)
                if v.dim() == 2:
                    v = v.unsqueeze(0)
                t_list.append(t)
                v_list.append(v)
            text_seq = torch.cat(t_list, dim=0)   # (B,Lt,D)
            img_seq  = torch.cat(v_list, dim=0)   # (B,Li,D)
        else:
            # Tensor / None 情况
            if text_emb is None and img_emb is None:
                text_seq = self.text_fallback   # (1,1,D)
                img_seq  = self.img_fallback    # (1,1,D)
            elif text_emb is None:
                if img_emb.dim() == 2:
                    img_emb = img_emb.unsqueeze(0)
                B = img_emb.size(0)
                text_seq = self.text_fallback.expand(B, 1, -1)
                img_seq  = img_emb
            elif img_emb is None:
                if text_emb.dim() == 2:
                    text_emb = text_emb.unsqueeze(0)
                B = text_emb.size(0)
                img_seq  = self.img_fallback.expand(B, 1, -1)
                text_seq = text_emb
            else:
                if text_emb.dim() == 2:
                    text_emb = text_emb.unsqueeze(0)
                if img_emb.dim() == 2:
                    img_emb = img_emb.unsqueeze(0)
                text_seq = text_emb
                img_seq  = img_emb

        # ---------- Cross-Attn: text_seq + img_seq → 32 个 multimodal tokens ----------
        fused_mm = self.cross_attn(text_seq, img_seq)    # (B, num_queries, fusion_dim)

        # ---------- 投到 LLM embedding 维度 ----------
        return self.proj_to_llm(fused_mm)                # (B, num_queries, llm_hidden_dim)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        text_emb=None,
        img_emb=None,
        hist_text_emb=None,
        hist_img_emb=None,
        **kwargs
    ):
        attn_mask = attention_mask
        input_embeds = self.llm.get_input_embeddings()(input_ids)

        # 新物品 + 历史 5 槽
        newitem_emb = self.fuse_item(text_emb, img_emb)  # (B,4,D)
        hist_embs = []
        for ht, hi in zip(hist_text_emb, hist_img_emb):
            hist_embs.append(self.fuse_item(ht, hi))     # list of (B,4,D)

        B, L, D = input_embeds.shape
        new_input_embeds, new_attn, new_labels = [], [], []
        pid = self.placeholder_token_ids
        for b in range(B):
            seq_emb, seq_mask, seq_label = [], [], []
            for i in range(L):
                tok_id = input_ids[b, i].item()
                if tok_id == pid["<NEWITEM>"]:
                    seq_emb.append(newitem_emb[b])
                    seq_mask.append(torch.ones(self.prompt_expand_len, device=attn_mask.device, dtype=attn_mask.dtype))
                    if labels is not None:
                        seq_label.append(torch.full((self.prompt_expand_len,), -100, device=attn_mask.device, dtype=torch.long))
                    continue

                replaced = False
                for name, tid in pid.items():
                    if name.startswith("<HIST") and tok_id == tid:
                        idx = int(name[5]) - 1
                        if idx < len(hist_embs):
                            seq_emb.append(hist_embs[idx][b])
                            seq_mask.append(torch.ones(self.prompt_expand_len, device=attn_mask.device, dtype=attn_mask.dtype))
                            if labels is not None:
                                seq_label.append(torch.full((self.prompt_expand_len,), -100, device=attn_mask.device, dtype=torch.long))
                        replaced = True
                        break

                if not replaced:
                    seq_emb.append(input_embeds[b, i:i+1, :])
                    seq_mask.append(attn_mask[b, i:i+1])
                    if labels is not None:
                        seq_label.append(labels[b, i:i+1])

            seq_emb  = torch.cat(seq_emb,  dim=0)
            seq_mask = torch.cat(seq_mask, dim=0)
            seq_lab  = torch.cat(seq_label, dim=0) if labels is not None else None

            new_input_embeds.append(seq_emb)
            new_attn.append(seq_mask)
            new_labels.append(seq_lab)

        max_len = max(e.size(0) for e in new_input_embeds)
        padded_embeds, padded_masks, padded_labels = [], [], []
        for e, m, l in zip(new_input_embeds, new_attn, new_labels):
            pad_len = max_len - e.size(0)
            padded_embeds.append(torch.cat([e, torch.zeros(pad_len, D, device=e.device, dtype=e.dtype)], dim=0))
            padded_masks.append(torch.cat([m, torch.zeros(pad_len, device=m.device, dtype=m.dtype)], dim=0))
            if labels is not None:
                padded_labels.append(torch.cat([l, torch.full((pad_len,), -100, device=m.device, dtype=torch.long)], dim=0))

        input_embeds = torch.stack(padded_embeds, dim=0)
        attn_mask    = torch.stack(padded_masks, dim=0)
        labels       = torch.stack(padded_labels, dim=0) if labels is not None else None

        if labels is not None:
            return self.llm(inputs_embeds=input_embeds, attention_mask=attn_mask, labels=labels)
        else:
            return self.llm(inputs_embeds=input_embeds, attention_mask=attn_mask)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.llm.prepare_inputs_for_generation(*args, **kwargs)