# 快速检查：1024是否存在
import numpy as np
codes = np.load("/home/users/nus/e1582377/RawMed/outputs_rqvae/12h/generated/train_RQVAE_indep/mimiciv_hi_code.npy", mmap_mode='r')
print(f"1024是否存在: {1024 in np.unique(codes[:100])}")
print(f"唯一值范围: {np.unique(codes[:100]).min()} - {np.unique(codes[:100]).max()}")