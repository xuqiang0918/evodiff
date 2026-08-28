# SDAA 适配说明（EvoDiff）

本文件记录 Microsoft [EvoDiff](https://github.com/microsoft/evodiff)（蛋白质序列离散扩散生成模型）从 NVIDIA CUDA 迁移到 Teco SDAA 加速卡的适配过程：环境、改动、配置、结果。

---

## 1. 适配环境

| 项目 | 版本/位置 |
| ---- | ------ |
| 硬件 | 太初 SDAA 卡（`device_count()==32`，`/dev/tcaicard0`） |
| 操作系统 | Loongnix Server 23.1（loongarch64） |
| PyTorch | 2.7.1 |
| Torch-SDAA | 3.2.0 |
| Python | 3.11（容器系统 python，已 `pip install evodiff==1.1.2`） |
| 模型 | OA_DM_38M（ unconditional 生成，38M 参数 fp32） |
| CUDA 基线 | 4×NVIDIA A100-PCIE-40GB |

> 关键事实：Teco 的 PyTorch 构建里 `torch.cuda` **存在但 `is_available()==False`**，真正的加速器由
> `torch.sdaa` 暴露（`is_available()==True`）。因此原代码里所有 `torch.cuda.is_available()` 判断
> 在 SDAA 上都会错误地落到 CPU——所有设备选择必须显式探测 `torch.sdaa`（见第 2 节）。

---

## 2. 适配策略

官方生成入口（`evodiff/generate.py` / `evodiff/conditional_generation.py`）在 main 入口处做**设备探测**，
按 **sdaa → cuda → cpu** 顺序：

- 优先 `hasattr(torch, 'sdaa')` 且 `torch.sdaa.is_available()` → `sdaa:<gpus>`
- 其次 `torch.cuda.is_available()` → `cuda:<gpus>`
- 兜底 `cpu`

探测到的 device 显式传入各生成函数（`generate_oaardm` / `inpaint` 等），函数默认参数保持官方原值
（`'cuda'` / `'gpu'`）不变——**CUDA 与 SDAA 双兼容**：同一份代码在两类机器上均可直接运行。

---

## 3. 改动清单

### 3.1 修改文件（均为 main 入口设备探测，带 `[sdaa-adapt]` 标记）

| 文件 | 修改 |
| ---- | ---- |
| `evodiff/generate.py` | main：`torch.cuda.set_device` → 探测式（sdaa/cuda/cpu 分发）；函数默认参数保持官方 `'cuda'` |
| `evodiff/conditional_generation.py` | main：`model.eval().cuda()` 与 set_device → 探测式；函数默认参数保持官方（scaffold `'gpu'`、inpaint `'cuda'`） |

### 3.2 新增文件（sdaa/ 目录）

| 文件 | 用途 |
| ---- | ---- |
| `sdaa/scripts/run_evodiff_min.py` | 无数据集依赖的快速验证脚本：gen（无条件生成，测吞吐/显存/产出 fasta）+ ppl（困惑度，直接吃 fasta，逻辑与官方 `analysis/sequence_perp.py` 一致） |
| `sdaa/scripts/run_benchmark.sh` | 完整基线：数值类型 / 4 长度性能 / GEN_N 条生成 / ppl |
| `sdaa/SDAA_ADAPTATION.md` | 本文档 |
| `sdaa/patches/` | git format-patch 补丁归档 |

### 3.3 未适配部分

- **MSA 模式**（`generate_msa.py` / `conditional_generation_msa.py`，MSA_OA_AR_DM 模型）尚未适配——依赖 OpenFold MSA 数据库，权重与数据未就绪，未验证。

---

## 4. 运行前配置

```bash
# ① SDAA 运行时（容器内 bashrc 已含）
source /opt/tecoai/setvars.sh

# ② 模型权重：OA_DM_38M 权重 tar 放入 TORCH_HOME
export TORCH_HOME=/data/application/xuqiang/evodiff_transfer
#   位置：$TORCH_HOME/hub/checkpoints/oaar-38M.tar

# ③ Python 依赖
pip install evodiff==1.1.2
```

> 权重 `oaar-38M.tar` 来源：HuggingFace `microsoft/evodiff`（首次运行自动下载，内网环境手动下载后放入上述路径）。

---

## 5. 兼容性说明

EvoDiff 推理为标准 torch 算子（Embedding / Linear / LayerNorm / softmax / multinomial），**未发现
SDAA 算子 dtype 覆盖缺口**。已通过全链路检查确认：模型参数、输入、输出、multinomial 采样全部在
SDAA 卡上，无 CPU fallback。

---

## 6. 已知限制与风险

1. **长序列生成性能差距扩大**：OA-DM 生成机制为 L 次串行 forward（机制性 O(L)）；CUDA 上单步
   forward 耗时随 L 基本持平，SDAA 上随 L 增长（128→1024 涨 42%），导致整体差距从 2.4×（L=128）
   扩大到 6.1×（L=1024）。
2. **MSA 模式未适配**（见 3.3）。
3. **driver api / driver version 不一致 warning**：运行时打印，不影响功能。
4. 官方 `generate.py` 的 baseline 对比绘图依赖 UniRef50 数据集；`run_evodiff_min.py` 不依赖。

---

## 7. 性能

> OA_DM_38M fp32，batch=1，无条件生成；SDAA 为龙芯单卡；CUDA 基线为 A100-40GB。

### 7.1 无条件生成吞吐（num_seqs=1）

| seq_len | SDAA 耗时 | SDAA 吞吐 | CUDA 耗时 | 性能比 |
| ------- | -------- | --------- | -------- | ------ |
| 128 | 4.12 s（warmup 后 3.10 s） | 31.1 aa/s（warmup 后 41.2） | 1.72 s | 2.4× |
| 256 | 8.70 s | 29.4 aa/s | — | — |
| 512 | 19.08 s | 26.8 aa/s | — | — |
| 1024 | 45.00 s | 22.8 aa/s | 7.42 s | 6.1× |

### 7.2 峰值显存与单步 forward

| seq_len | 峰值显存（torch 侧） | 单次 forward |
| ------- | ------------------ | ------------ |
| 128 | 151 MB | 25.7 ms |
| 256 | 153 MB | 27.7 ms |
| 512 | 158 MB | 29.8 ms |
| 1024 | 170 MB | 36.5 ms |

---

## 附：验证记录

**环境验证（实测记录）**

- `torch.sdaa` 正确识别（32 卡，`/dev/tcaicard0`）；
- 全链路设备检查：模型参数 / 输入 / 输出 / multinomial 采样全部位于 `sdaa:0`，无 CPU fallback；
- `py_compile` 已于本次适配改造后执行通过（`generate.py` / `conditional_generation.py`）；
- 改造后冒烟：gen 128 → 3.10 s / 41.2 aa/s / peak 151 MB，功能正常。

**适配命令**

```bash
# 官方无条件生成 CLI（<repo> 为仓库根目录）
source /opt/tecoai/setvars.sh
export TORCH_HOME=<权重目录>
cd <repo>
python evodiff/generate.py --model-dir oa_ar_dsm_38M --num-seqs 100 --seq-len 256

# 辅助基线（性能/显存/生成/ppl 一步到位）
bash sdaa/scripts/run_benchmark.sh              # 默认生成 100 条
GEN_N=1000 bash sdaa/scripts/run_benchmark.sh   # 1000 条
```
