# HuggingFace 上传教程（全部从本地上传）

最终会在 HuggingFace 上得到两个仓库：

| 仓库 | 类型 | 内容 | 大小 |
|---|---|---|---|
| `你的用户名/mmrev-data` | Dataset | 三个类目 × 三个场景的 train/test、原始 `final.jsonl`、split、图片描述、teacher scores | 约 450 MB |
| `你的用户名/mmrev-llama3.1-8b-grpo` | Model | GRPO checkpoint（step 3147）的 LoRA adapter + tokenizer | 约 16 MB |

---

## 第 0 步：准备环境

需要 Python 3.9+。在终端运行：

```bash
pip install -U huggingface_hub numpy
```

装好后确认命令可用：

```bash
hf --help
```

> 本教程使用新版 `hf` 命令（huggingface_hub ≥ 1.0）。如果你的环境只有旧版 `huggingface-cli`，对照表在最后。

---

## 第 1 步：注册账号并生成 Token

1. 打开 https://huggingface.co/join 注册。**用户名会出现在所有链接里**（例如 `huggingface.co/datasets/chuqiao/mmrev-data`），建议用简短、正式的名字。
2. 登录后，右上角头像 → **Settings** → 左侧 **Access Tokens** → **Create new token**。
3. Token type 选 **Write**，名字随便写（如 `mmrev-upload`）→ Create。
4. 复制以 `hf_` 开头的 token。**它只显示一次**，先存好；不要把它写进任何代码或提交到 GitHub。

---

## 第 2 步：本地登录

```bash
hf auth login
```

按提示粘贴 token（粘贴时屏幕不显示，这是正常的），回车。然后验证：

```bash
hf auth whoami
```

显示你的用户名即表示登录成功。

---

## 第 3 步：把本地文件放进一个工作目录

新建一个文件夹（例如 `mmrev_hf/`），按下面的结构放好你已经下载的文件。文件夹名可以不同，第 4 步命令里对应改掉即可。

```
mmrev_hf/
├── hf_kit/                       ← 本工具包（4 个文件）
│   ├── prepare_hf.py
│   ├── model_card.md
│   ├── dataset_card.md
│   └── HF_UPLOAD_GUIDE.md
├── final/                        ← 解压 final.zip 得到的文件
│   ├── final_Arts.jsonl
│   ├── final_Sports.jsonl
│   ├── final_Toys.jsonl
│   └── ...（其他类目的文件可以留着，脚本只取 Arts/Sports/Toys）
├── split/                        ← 解压 split.zip
│   ├── Arts_both_split.npz
│   └── ...（共 9 个 .npz）
├── img2txt/                      ← 解压 img2txt.zip，以及其他图片描述文件
│   ├── train_all_hist_both_img2txt_Arts.jsonl
│   └── ...
├── per_sample/                   ← 两个 CSV
│   ├── per_sample_detail_Arts_train_smooth_bleu.csv
│   └── per_sample_detail_Toys_train_smooth_bleu.csv
└── checkpoint-3147/              ← 解压 checkpoint-3147.zip
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...
```

**关于 `img2txt/`**：脚本会自动收录这个文件夹下所有文件名（或上级文件夹名）里带 `Arts` / `Sports` / `Toys` 的 `.jsonl`。如果之后从 HPC 拿到了更多图片描述文件（见本文末"建议补充的文件"），直接放进来、重跑第 4 步即可。

---

## 第 4 步：生成上传文件夹

在 `mmrev_hf/` 目录下运行（把 `你的用户名` 换成第 1 步注册的用户名）：

```bash
python hf_kit/prepare_hf.py \
    --final_dir      final \
    --split_dir      split \
    --img2txt_dir    img2txt \
    --teacher_dir    per_sample \
    --checkpoint_dir checkpoint-3147 \
    --hf_user        你的用户名 \
    --out            hf_upload
```

脚本会做这些事：

- 按 split 把每个类目 × 场景的 train/test 抽成独立的 `.jsonl`，字段统一、可直接被 `load_dataset` 读取；
- 额外生成每个类目的 `<类目>_mix_split.npz`（三场景并集，即 mix-SFT 与 GRPO 训练所用的划分）；
- 原始 `final.jsonl`、`.npz`、图片描述、teacher scores 原样放进对应目录；
- 从 checkpoint 中只保留推理需要的文件，丢弃 `optimizer.pt` 等训练状态；
- 把 `adapter_config.json` 里的集群路径 `/scratch/xs2334/models/llama3.1-8b` 改成 `meta-llama/Llama-3.1-8B`，否则别人无法加载；
- 写入两份说明页（`README.md`），并替换其中的用户名。

运行结束会打印每个场景的样本数和每个文件的大小。正确的样本数应为：

```
Arts    both: 750/150   text_only: 707/141   img_only: 641/128
Sports  both: 898/180   text_only: 891/178   img_only: 840/168
Toys    both: 1216/243   text_only: 1190/238   img_only: 1230/246
```

---

## 第 5 步：检查说明页

用任意编辑器打开下面两个文件，确认内容，并修改仍是占位符的地方：

- `hf_upload/mmrev-data/README.md`
- `hf_upload/mmrev-llama3.1-8b-grpo/README.md`

需要你手动改的：

- `technical report, link TBA` → arXiv 发布后换成链接（现在可以先保留，之后再改）；
- 数据集说明页底部 BibTeX 里的 `author = {Chen, Chuqiao and others}` → 改成完整作者名单；
- GitHub 链接 `https://github.com/你的用户名/mmrev`：如果 GitHub 和 HuggingFace 用户名不同，改成 GitHub 上的实际地址。

（可选）上传前在本地测试数据能否被读取：

```bash
pip install datasets
python -c "from datasets import load_dataset; import os; print(load_dataset(os.path.abspath('hf_upload/mmrev-data'), 'arts_both'))"
```

---

## 第 6 步：上传

`hf upload` 会在仓库不存在时自动创建，默认是**公开**仓库。

**上传数据集：**

```bash
hf upload 你的用户名/mmrev-data hf_upload/mmrev-data . \
    --repo-type dataset \
    --commit-message "Initial release: MMRev data"
```

约 450 MB，取决于网速可能需要几分钟到半小时。**中途断开的话，重新运行同一条命令即可续传**。

**上传模型：**

```bash
hf upload 你的用户名/mmrev-llama3.1-8b-grpo hf_upload/mmrev-llama3.1-8b-grpo . \
    --commit-message "Initial release: mix-SFT -> GRPO LoRA adapter (step 3147)"
```

> 想先私下检查、确认无误再公开：在两条命令后都加上 `--private`，检查完后到网页上 Settings → Change repository visibility 改成 Public。

---

## 第 7 步：在网页上检查

**数据集**：打开 `https://huggingface.co/datasets/你的用户名/mmrev-data`

- 页面上方的 **Dataset Viewer** 应出现 9 个子集（`arts_both`、`arts_text_only` …），每个都有 train/test。Viewer 第一次生成需要几分钟，刚上传时显示 "processing" 是正常的；
- **Files and versions** 标签页里可以看到 `raw/`、`splits/`、`image_descriptions/`、`teacher_scores/`。

**模型**：打开 `https://huggingface.co/你的用户名/mmrev-llama3.1-8b-grpo`

- 说明页右侧会显示 base model 为 `meta-llama/Llama-3.1-8B`，并带有 `peft` 标签；
- Files 里应只有 7 个文件（不含 `optimizer.pt`）。

最后在任意装有 `datasets` 的机器上验证远程读取：

```bash
python -c "from datasets import load_dataset; print(load_dataset('你的用户名/mmrev-data', 'arts_both'))"
```

---

## 第 8 步：之后更新

修改了说明页、加了新的图片描述文件等，重新运行第 4 步生成文件夹，再运行第 6 步同样的命令即可。只会上传有变化的文件，并在 Files and versions 里留下新的提交记录。

只改一个文件也可以单独上传，例如：

```bash
hf upload 你的用户名/mmrev-data hf_upload/mmrev-data/README.md README.md --repo-type dataset
```

---

## 关于公开与授权

- **放进 CV 必须是 Public**，否则别人打开是 404。
- 模型基于 Llama 3.1，说明页已声明遵循 Llama 3.1 Community License。基座模型 `meta-llama/Llama-3.1-8B` 本身需要在 HuggingFace 上申请访问权限；你的 adapter 可以公开，但别人要用它，需要先获得基座模型的权限。这是正常情况，不需要额外处理。
- 数据集派生自 Amazon Reviews 2023，说明页已注明遵循原数据集的使用条款、仅用于研究。

---

## 建议补充的文件（为了完整复现）

你目前上传的图片描述只有 `train_all_hist_both_img2txt`（三个类目各一份）。训练和评测代码还会读取以下文件，路径来自你的推理脚本：

```
/scratch/xs2334/DescGen/output/image_descriptions/{Arts,Sports,Toys}/
    train_all_hist_img_only_img2txt.jsonl
    test_all_hist_both_img2txt.jsonl
    test_all_hist_img_only_img2txt.jsonl
    hist_img2txt.jsonl              ← 历史商品的图片描述（如果存在）
```

拿到后放进 `img2txt/`（建议按类目建子文件夹，或在文件名末尾加 `_Arts` 等），重跑第 4、6 步即可。缺少这些文件不影响上传，但别人无法完整复现结果。**尤其是 `img_only` 的图片描述文件**：训练代码靠它判断哪些样本属于 img_only 场景，缺了它，img_only 样本会被当成 text_only 处理。

另外 `per_sample_detail` 目前只有 Arts 和 Toys，若有 Sports 的也可以一起放进 `per_sample/`。

---

## 附：新旧命令对照

| 新版 `hf` | 旧版 `huggingface-cli` |
|---|---|
| `hf auth login` | `huggingface-cli login` |
| `hf auth whoami` | `huggingface-cli whoami` |
| `hf repos create 名字 --repo-type dataset` | `huggingface-cli repo create 名字 --type dataset` |
| `hf upload 名字 本地路径 . --repo-type dataset` | `huggingface-cli upload 名字 本地路径 . --repo-type dataset` |

也可以完全不用命令行，用 Python：

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(repo_id="你的用户名/mmrev-data", folder_path="hf_upload/mmrev-data", repo_type="dataset")
api.upload_folder(repo_id="你的用户名/mmrev-llama3.1-8b-grpo", folder_path="hf_upload/mmrev-llama3.1-8b-grpo")
```
