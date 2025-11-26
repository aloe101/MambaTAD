## Usage

Before running the TAD experiments, you need to install the mamba module.
```bash
# install mamba
cd causal-conv1d
python setup.py develop
cd ..
cd mamba
python setup.py develop
cd ..
```


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train VideoMambaSuite on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/mambatad/thumos_internvideo6b.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test VideoMambaSuite on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/mambatad/thumos_internvideo6b.py --checkpoint path/to/your/ckpt
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@misc{2024videomambasuite,
      title={Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding}, 
      author={Guo Chen, Yifei Huang, Jilan Xu, Baoqi Pei, Zhe Chen, Zhiqi Li, Jiahao Wang, Kunchang Li, Tong Lu, Limin Wang},
      year={2024},
      eprint={2403.09626},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```