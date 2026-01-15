from flash_attn_interface import  flash_attn_varlen_func, flash_attn_func
import torch
import triton
import math
import deep_gemm
from benchmark.quant_kernel import act_quant
from flash_mla import flash_mla_sparse_fwd

dsa_qk_dim = 576
dsa_v_dim = 512
dsa_head_q =128
dsa_head_k = 1

index_head_q = 64
index_head_k = 1
index_qk_dim = 128

mla_qk_dim = 192
mla_v_dim = 192
mla_head_q = 128
mla_head_kv = 128

dytpe = torch.bfloat16
device = "cuda:0"

B = 2


def generate_rowwise_permutations(M,N):
    noise = torch.rand(M, N, device=device)
    result = torch.argsort(noise, dim=1)
    return result.to(torch.int32)

def mock_data(B, S, topK):
    
    BS = B * S
    
    dsa_q = torch.randn(BS, dsa_head_q,dsa_qk_dim, dtype=dytpe, device=device)
    dsa_k = torch.randn(BS, dsa_head_k,dsa_qk_dim, dtype=dytpe, device=device)
    
    index_q = torch.randn(BS, index_head_q, index_qk_dim, dtype=dytpe, device=device)
    index_k = torch.randn(BS, index_head_k, index_qk_dim, dtype=dytpe, device=device)
    
    mla_q = torch.randn(BS, mla_head_q, mla_qk_dim, dtype=dytpe, device=device)
    mla_k = torch.randn(BS, mla_head_kv, mla_qk_dim, dtype=dytpe, device=device)
    mla_v = torch.randn(BS, mla_head_kv, mla_v_dim, dtype=dytpe, device=device)
    
    
    seq_offsets = torch.arange(0, (B+1) *S, S, dtype=torch.int32, device=device)
    
    weights = torch.randn(BS, index_head_q, dtype=torch.float32, device=device)
    
    # generate random index
    index = generate_rowwise_permutations(B*S, topK)
    
    return dsa_q, dsa_k,index_q,index_k, mla_q,mla_k, mla_v, seq_offsets, weights, index



def gen_mqk_start_end(offsets, lengths):
    # 1. 生成 start 向量
    # 使用 repeat_interleave 将 offsets[0:-1] 按照 lengths 重复
    starts = torch.repeat_interleave(offsets[:-1], lengths)
    
    # 2. 生成 end 向量 (0, 1, 2, 3, 4)
    # 先生成一个从 0 到总长度的序列
    total_len = offsets[-1] - offsets[0]
    ends = torch.arange(total_len, device=offsets.device)
    
    return starts.to(torch.int32), ends.to(torch.int32)


def dsa_fwd(q, k, index_q,index_k,seq_offsets, topk, fp8_weights, max_seqlen_k, index):
    
    total_seqlen_q, head_q, dim_q = q.shape
    total_seqlen_kv, head_kv, dim_k = k.shape
    assert total_seqlen_q == total_seqlen_kv
    assert head_kv == 1
    assert dim_q == dim_k
    assert fp8_weights.shape[0] == total_seqlen_q
    assert fp8_weights.shape[1] == index_q.shape[1]
    
    
    q_fp8, q_scale = act_quant(index_q)
    k_fp8, k_scale = act_quant(index_k)
    q_scale = q_scale.squeeze(-1)
    k_scale = k_scale.squeeze(-1)
    
    k_fp8 = k_fp8.squeeze(1)
    assert q_scale.shape == fp8_weights.shape, f"q_scale and fp8_weights must have the same shape, but got {q_scale.shape} and {fp8_weights.shape}"
    
    
    lengths = seq_offsets[1:] - seq_offsets[:-1]
    lengths = lengths.to(dtype=torch.int32, device=q.device)
    
    # generate start and end index for causal
    ks, ke = gen_mqk_start_end(seq_offsets, lengths)
    logits = deep_gemm.fp8_mqa_logits(q_fp8, (k_fp8, k_scale), q_scale * fp8_weights, ks, ke, max_seqlen_k=max_seqlen_k, clean_logits=False)
    
    # convert indexes 2 ragged index
    ragged_indexes  = index + (torch.repeat_interleave(seq_offsets[:-1], lengths))[:, None]
    sm_scale = 1 / math.sqrt(dsa_qk_dim)
    
    # sparse attention
    res = flash_mla_sparse_fwd( q , k, ragged_indexes[:,None,:], sm_scale=sm_scale)
    return (res[0], logits)



T_vals = [1024 * 2**i for i in range(0, 6)]
topK_vals = [2048]

param_combinations = []
for T in T_vals:
    for K in topK_vals:
        if K < T:
            param_combinations.append((T, K))
        
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T", "topK"],
        x_vals=param_combinations,
        line_arg="provider",
        line_vals=["mla", "dsa"],
        line_names=["mla", "dsa"],
        styles=[
            ("green", "-"),
            ("green", "solid"),
            ("blue", "-"),
            ("blue", "solid"),
            ("red", "-"),
            ("green", "dotted"),
        ],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, topK, provider):
    
    
    dsa_q, dsa_k, index_q,index_k,mla_q, mla_k, mla_v, seq_offsets, weights, index = mock_data(B, T, topK)
    print(f"B: {B}, T: {T}, topK: {topK}")

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == "dsa":
        results = triton.testing.do_bench(
            lambda: dsa_fwd(
                dsa_q,
                dsa_k,
                index_q,
                index_k,
                seq_offsets,
                topK,
                weights,
                T,
                index
            ),
            quantiles=quantiles,
        )
    elif provider == "mla":
        results = triton.testing.do_bench(
            lambda: flash_attn_varlen_func(
                mla_q,
                mla_k,
                mla_v,
                seq_offsets,
                seq_offsets,
                T,
                T,
                causal= True
            ),
            quantiles=quantiles,
        )

    return results

if __name__ == "__main__":
    
    torch.set_default_device(device)
    with torch.cuda.device(device):
        benchmark.run(print_data=True, save_path=".")