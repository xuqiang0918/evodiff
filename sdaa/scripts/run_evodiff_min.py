"""
EvoDiff CUDA 快速验证脚本（不依赖 UniRef50 数据集）
===================================================
模式：
  gen  无条件生成（全 mask 逐位置解算）—— 测吞吐 + 峰值显存，产出生成样本
  ppl  困惑度评测（随机 mask + OAMaskedCrossEntropyLoss）—— 对已知序列测拟合度
       逻辑与官方 analysis/sequence_perp.py 的 sum_nll_mask 完全一致，只是不依赖 UniRef50，
       直接输入任意真实蛋白序列（默认内置 1CRN crambin，或 --fasta 指定）

用法：
  python run_evodiff_min.py --mode gen --seq-len 128 --num-seqs 1 --gpu 0
  python run_evodiff_min.py --mode ppl --gpu 0
  python run_evodiff_min.py --mode ppl --fasta my_seqs.fasta --gpu 0
"""
import argparse
import time

import numpy as np
import torch

from evodiff.pretrained import OA_DM_38M
from evodiff.collaters import OAMaskCollater
from evodiff.losses import OAMaskedCrossEntropyLoss

# 1CRN crambin（真实蛋白，用作默认 ppl 测试序列）
DEFAULT_SEQ = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"


def read_fasta(path):
    seqs = []
    cur = ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append(cur)
                cur = ""
            else:
                cur += line.upper()
    if cur:
        seqs.append(cur)
    return seqs


def run_gen(model, tokenizer, seq_len, batch, device):
    """全 mask 无条件生成，与官方 generate_oaardm 逻辑一致。"""
    all_aas = tokenizer.all_aas
    mask_id = tokenizer.mask_id

    sample = torch.full((batch, seq_len), mask_id, dtype=torch.long, device=device)
    loc = np.arange(seq_len)
    np.random.shuffle(loc)
    timestep = torch.zeros(batch, dtype=torch.long, device=device)

    torch.sdaa.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for i in loc:
            pred = model(sample, timestep)                  # (B, L, vocab)
            p = pred[:, i, : len(all_aas) - 6]              # 只取标准 AA 子集
            p = torch.softmax(p, dim=1)
            p_sample = torch.multinomial(p, num_samples=1)
            sample[:, i] = p_sample.squeeze(1)
    dt = time.time() - t0
    peak_mb = torch.sdaa.max_memory_allocated() / 1024 / 1024

    for s in sample:
        print(">generated_sequence")
        print(tokenizer.untokenize(s))
    print(
        f"[perf] seq_len={seq_len} batch={batch} "
        f"time={dt:.2f}s speed={seq_len * batch / dt:.1f} aa/s "
        f"peak_mem={peak_mb:.0f} MB"
    )


def run_ppl(model, collater, tokenizer, seqs, device):
    """对已知序列算困惑度（随机 mask 比例 ~50%，与官方评测一致）。"""
    loss_func = OAMaskedCrossEntropyLoss(reweight=False)
    total_nll = 0.0
    total_tokens = 0
    for seq in seqs:
        seq = seq.strip()
        if len(seq) < 10:
            continue
        # 注意：Tokenizer.tokenize 期望嵌套结构（seq[0] 为完整序列），
        # 官方传 (seq, idx) 元组列表；此处保持同口径
        src, timestep, tgt, mask = collater([(seq, 0)])
        timestep = torch.tensor([0] * len(src))              # mask 模型不用真实 timestep
        input_mask = (src != tokenizer.pad_id).float()
        src = src.to(device)
        timestep = timestep.to(device)
        tgt = tgt.to(device)
        mask = mask.to(device)
        input_mask = input_mask.to(device)
        with torch.no_grad():
            outputs = model(src, timestep)
        _, nll_loss = loss_func(outputs[:, :, :], tgt, mask, timestep, input_mask)
        total_nll += nll_loss.item()
        total_tokens += mask.sum().item()
        print(f"[ppl] seq_len={len(seq)} masked={mask.sum().item()} nll={nll_loss.item():.2f}")
    perp = np.exp(total_nll / total_tokens)
    print(f"[ppl] sequences={len(seqs)} total_tokens={total_tokens} PERPLEXITY={perp:.3f}")
    return perp


def main():
    parser = argparse.ArgumentParser(description="EvoDiff CUDA min test")
    parser.add_argument("--mode", choices=["gen", "ppl"], default="gen",
                        help="gen=无条件生成(性能基线); ppl=困惑度评测(精度基线)")
    parser.add_argument("--seq-len", type=int, default=128, help="gen 模式：生成序列长度")
    parser.add_argument("--num-seqs", type=int, default=1, help="gen 模式：batch 大小")
    parser.add_argument("--fasta", type=str, default=None, help="ppl 模式：输入 FASTA 文件（默认用内置 crambin）")
    parser.add_argument("--gpu", type=int, default=0, help="SDAA device id")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device(f"sdaa:{args.gpu}")
    torch.sdaa.set_device(args.gpu)
    print(f"[info] device = {device}  sdaa_available = {torch.sdaa.is_available()}")

    model, collater, tokenizer, scheme = OA_DM_38M()
    model = model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[info] model = OA_DM_38M  params = {n_params:.1f}M  scheme = {scheme}")

    if args.mode == "gen":
        run_gen(model, tokenizer, args.seq_len, args.num_seqs, device)
    else:
        seqs = read_fasta(args.fasta) if args.fasta else [DEFAULT_SEQ]
        if not seqs:
            raise SystemExit("未读取到任何序列，请检查 --fasta 文件")
        run_ppl(model, collater, tokenizer, seqs, device)


if __name__ == "__main__":
    main()
