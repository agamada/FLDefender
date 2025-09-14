import torch
import time

# 大规模参数
N, d = 1000, 500  # 1000个客户端，每个更新500维

all_updates = torch.randn(N, d, device="cpu")  # 放CPU避免显存直接爆
all_updates = all_updates.double()
# 方法1：广播方式
start = time.time()
distances1 = torch.sum((all_updates[:,None,:] - all_updates[None,:,:]) ** 2, dim=2)
end = time.time()
print("方法1: 时间 {:.4f}s, 结果大小 {}".format(end-start, distances1.shape))

# 方法2：代数公式
start = time.time()
norms = torch.sum(all_updates ** 2, dim=1, keepdim=True)
distances2 = norms + norms.T - 2 * all_updates @ all_updates.T
end = time.time()
print("方法2: 时间 {:.4f}s, 结果大小 {}".format(end-start, distances2.shape))

# 验证结果一致
print("结果是否完全相等？", torch.allclose(distances1, distances2, atol=1e-6))
diff = (distances1 - distances2).abs()
print("最大绝对误差: ", diff.max().item())
print("最大相对误差: ", (diff / (distances2.abs() + 1e-12)).max().item())

