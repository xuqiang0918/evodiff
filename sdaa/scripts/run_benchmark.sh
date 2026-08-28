#!/bin/bash
# ============================================================
# EvoDiff OA_DM_38M SDAA 完整基线脚本（无条件生成优先）
# 用法:
#   bash run_benchmark.sh                        # 生成 100 条（快）
#   GEN_N=1000 bash run_benchmark.sh             # 生成 1000 条（分布统计级）
# 输出: 数值类型 / 4 长度性能 / GEN_N 条生成 / ppl 精度
# 环境: abb3-sdaa 容器（loongarch64, torch 2.7.1 + torch_sdaa 3.2.0）
#       需先 source /opt/tecoai/setvars.sh（交互式 shell 的 .bashrc 已自动加载）
# ============================================================
set -e
cd /data/application/xuqiang/evodiff/sdaa
export TORCH_HOME=/data/application/xuqiang/evodiff_transfer
PY="${PY:-python}"                       # 容器系统 python（已装 torch_sdaa + evodiff）
RUN=scripts/run_evodiff_min.py           # SDAA 适配版脚本（scripts/ 下）
GEN_N="${GEN_N:-100}"      # 生成条数，可用环境变量覆盖
GEN_LEN="${GEN_LEN:-256}"  # 生成长度

echo "========== [0] 数值类型（fp32?） =========="
$PY -c "
import torch
from evodiff.pretrained import OA_DM_38M
m,_,_,_ = OA_DM_38M()
m.eval().to('sdaa:0')
print('param dtypes :', set(p.dtype for p in m.parameters()))
print('params       : %.2f M (fp32 = %.1f MB)' % (sum(p.numel() for p in m.parameters())/1e6, 4*sum(p.numel() for p in m.parameters())/1024/1024))
x = torch.full((1, 128), 26, dtype=torch.long, device='sdaa:0')
t = torch.zeros(1, dtype=torch.long, device='sdaa:0')
with torch.no_grad():
    print('output dtype :', m(x, t).dtype)
" 2>&1 | grep -v pkg_resources

echo "========== [1] 单样本性能（warmup + 128/256/512/1024） =========="
$PY $RUN --mode gen --seq-len 128 --num-seqs 1 --gpu 0 2>/dev/null | grep perf   # warmup
for L in 128 256 512 1024; do
  $PY $RUN --mode gen --seq-len $L --num-seqs 1 --gpu 0 2>/dev/null | grep perf
done

echo "========== [2] 生成 ${GEN_N} 条（len ${GEN_LEN}） -> data/gen${GEN_N}.fasta =========="
$PY $RUN --mode gen --seq-len $GEN_LEN --num-seqs $GEN_N --gpu 0 2>/dev/null \
  | awk '/^>/{getline; print ">gen_" NR; print}' > data/gen${GEN_N}.fasta
echo "records: $(grep -c '^>' data/gen${GEN_N}.fasta)"

echo "========== [3] 精度（PERPLEXITY，越低越好） =========="
echo "--- 生成样本（${GEN_N} 条） ---"
$PY $RUN --mode ppl --fasta data/gen${GEN_N}.fasta --gpu 0 2>/dev/null | grep PERPLEXITY
#echo "--- 天然测试集（48,941 条） ---"
#$PY $RUN --mode ppl --fasta data/test.fasta --gpu 0 2>/dev/null | grep PERPLEXITY

echo "========== 完成 =========="
