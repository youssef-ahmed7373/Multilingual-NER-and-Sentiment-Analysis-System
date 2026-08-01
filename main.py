def main():
    import torch
    print(torch.cuda.is_available())      # should print True
    print(torch.cuda.get_device_name(0))  # should print "NVIDIA GeForce GTX 1650"


if __name__ == "__main__":
    main()
print("1")
import torch

print("2")

import transformers

print("3")

from transformers import pipeline

print("4")
