import torch, faiss
print("torch.cuda.is_available() =", torch.cuda.is_available())
try:
    print("faiss.get_num_gpus()   =", faiss.get_num_gpus())
except Exception as e:
    print("faiss GPU check error:", e)