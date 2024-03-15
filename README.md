# Swin Transformer 1D

This repo is the PyTorch implementation of ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"](https://arxiv.org/pdf/2103.14030.pdf)
for ***1-dimensional*** data such as audio signal.

The original codes are borrowed from the following repository:
- [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

## Summery

This repository contains :
- Swin Transformer for 1-dimensional data -> [ ***models.swin_transformer_1d.SwinTransformerLayer*** ]
- Swin Transformer V2 for 1-dimensional data -> [ ***models.swin_transformer_1d_v2.SwinTransformerV2Layer*** ]

## Difference from the original 2-d implementation

1. While, in the original repository, a `SwinTransformer` module that directly takes image data as input is provided, this repository only provides a general-purpose `SwinTransformerLayer` module that takes 1-dimensional embeddings as input.
1. For shifted-window Transformer, ***zero-padding shift*** is applied instead of cyclic shift
to accept input sequences of varying lengths during both training and inference time, and to get the benefit from fused implementation of the [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) module of PyTorch [***TODO***].


## How to use

`SwinTransformer(V2)Layer` can be used generically with any 1-dimensional embedding as input.

* NOTE : In case of `depth=2`, the layer corresponds to `Two Successive Swin Transformer Blocks` as described in the paper.

Here's an example code snippet demonstrating how to use it.
```python
device = 'cuda'
dim = 128
depth = 4
num_heads = 4
window_size = 12

BS = 10
L = window_size * 10
x = torch.rand(BS, L, dim).to(device)

swin_t = SwinTransformerLayer(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size)
swin_t = swin_t.to(device)

# forward
out = swin_t(x)

print(x.shape)  # (BS, L, dim)
print(out.shape)  # (BS, L, dim)
```

## Introduction

**Swin Transformer** (the name `Swin` stands for **S**hifted **win**dow) is initially described in [arxiv](https://arxiv.org/abs/2103.14030), which capably serves as a
general-purpose backbone for computer vision. It is basically a hierarchical Transformer whose representation is
computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention
computation to non-overlapping local windows while also allowing for cross-window connection.

The following figure dipicts a 2-d (original) Swin Transformer from the paper.

![teaser](assets/teaser.png)

## Citing Swin Transformer

```
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
## Citing Swin Transformer V2
```
@inproceedings{liu2021swinv2,
  title={Swin Transformer V2: Scaling Up Capacity and Resolution}, 
  author={Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
  booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
