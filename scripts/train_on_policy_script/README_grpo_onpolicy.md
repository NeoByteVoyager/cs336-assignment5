# Kaggle 双 T4：GRPO on-policy experiment

训练入口：`scripts/train_on_policy_script/train_on_policy_script.py`。

阅读作业 PDF §4.2–4.3 后修订的版本。只修改/新增 `scripts/` 和对应测试，不修改你的 `cs336_alignment` 实现；没有 monkey patch，不重写 GRPO loss。

## 直接在 Kaggle 运行

Notebook 选择 **GPU T4 x2**，打开 Internet。在 Notebook 中执行（仓库地址替换为你自己的）：

```python
!git clone <你的仓库地址> /kaggle/working/assignment5-alignment
%cd /kaggle/working/assignment5-alignment
# 已经 clone 时改用 !git pull
%pip install -r scripts/train_on_policy_script/requirements-grpo.txt
```

不要安装项目 `[gpu]` extra：其中含 FlashAttention/vLLM 的其他依赖，且项目 Python 版本限制不一定匹配 Kaggle。
基础依赖不安装 vLLM，保留 Kaggle 已有 CUDA torch（要求 >=2.4）；安装 transformers 后若 kernel 已导入旧版本，重启 kernel。
本脚本直接读取仓库现有 `data/gsm8k/train.jsonl` 和 `test.jsonl`。

先检查 GPU，再执行两步小规模验证：

```python
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.version.cuda); print([(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())])"
!bash scripts/train_on_policy_script/run_kaggle_grpo.sh smoke
```

默认输出位置：`/kaggle/working/grpo-smoke/`。应看到训练/验证指标、样例和 `checkpoint-000002/`。
两步检查使用 64 个生成 tokens，可能截断解题过程，只验证执行链路，不用于判断训练效果。

通过后运行 50 步体验：

```python
!bash scripts/train_on_policy_script/run_kaggle_grpo.sh experience
```

默认输出：`/kaggle/working/grpo-experience/`。
可以在末尾覆盖参数，例如缩小生成长度/验证集以节省时间：

```python
!bash scripts/train_on_policy_script/run_kaggle_grpo.sh experience --max-new-tokens 256 --eval-size 8
```

更改输出目录通过环境变量，而不是在 wrapper 末尾传 `--output-dir`，这样 console.log 与模型日志在同一目录：

```python
!GRPO_OUTPUT_DIR=/kaggle/working/grpo-seed1 bash scripts/train_on_policy_script/run_kaggle_grpo.sh experience --seed 1
```

Notebook 的 Save Version / Save & Run All 选择保存输出文件，或在 session 结束前下载输出目录。Git push/pull 只搬运代码，不会保存 Kaggle 运行产物。
不要将模型 checkpoint 提交到 Git。

## 双 GPU 与精度

一个 Python 训练进程，不用 DDP/FSDP，不用 `torchrun`：

- GPU 0：完整 FP32 policy、梯度、梯度检查点，默认训练微批次为 1。
- CPU：FP32 参数副本和 PyTorch AdamW 状态，betas=(0.9,0.95)，weight_decay=0。
- GPU 1：完整 FP32 rollout 模型，默认使用 Transformers `model.generate()`，小批量左 padding 生成。
- 每次采样前逐参数/缓冲区同步最新 policy；更新后的验证也先同步。每批 rollout 只调用一次现有 `grpo_train_step`，然后丢弃；下一步重新同步、重新采样。

为什么没有直接 FP16 Adam 或普通 GPU AdamW：
现有 `grpo_train_step` 内部直接 `.backward()` 和 `.step()`，无法从外部按标准方式插入 GradScaler。真实 OLMo 配置统计为 1,484,916,736 个参数：FP32 参数+梯度约 **11.06 GiB**，加两份 FP32 Adam 状态后约 **22.13 GiB**，单张 T4 放不下。
因此全程保持 FP32，把 AdamW 更新放在 CPU。两个 GPU 的显存不会自动合并成一个 32 GB 空间。

`TrainingOptimizer` 是显式 optimizer 子类：现有 train step 传入 `max_grad_norm=None`，在其最后调用 adapter.step() 时，由 adapter **记录裁剪前全局范数、裁剪一次、检查非有限梯度、调用 PyTorch AdamW、复制参数回 GPU**。
它逐参数传递 CPU 梯度，避免额外存储完整 CPU 梯度；未改 loss/backward。CPU 内存中参数副本+Adam 状态约 **16.60 GiB**，还有加载、保存和 CPU 临时张量开销，需足够主机内存（建议约 30 GiB 可用 RAM）。
CPU 更新和 PCIe 复制会慢，但比改 loss/引入分布式框架简单。可用 `--optimizer sgd` 做更轻的链路检查；这不再是作业建议的 AdamW 实验。

单卡调试可指定 `--rollout-gpu -1 --rollout-batch-size 1`。CPU 仅用于小模型离线测试：`--device cpu`，不建议真实 1B CPU 训练。

## T4 能不能使用 vLLM？

**硬件支持，但具体 Kaggle 软件栈需要实测。** 官方安装页列出 T4/compute capability 7.5；OLMo 2 在支持的架构列表中。
T4 不支持本方案需要的原生 BF16；不能沿用仓库 `vllm_utils.start_server()` 固定的 `--dtype bfloat16`。

本版保留默认 Transformers 路径（也符合最初不使用 vLLM 的要求），并提供显式可选 `--rollout-backend vllm`：

- 固定 vLLM 0.19.1，独立子进程运行 GPU 1；`float32`、`TRITON_ATTN`、`--enforce-eager`，不强制 FlashAttention 2。
- 与训练 policy 同为 FP32，避免 FP16 模型采样/FP32 policy 训练额外引入精度差异。
- 复用仓库 `init_weight_sync` / `sync_policy_weights` 的 NCCL 权重传输和缓存清理，不使用过期权重。
- 传入 token prompts，不自动加 BOS；同步后分批请求 completion，检查数量和 token/text 对齐。
- server 日志写入 `vllm-server.log`，启动失败不会悄悄回退至旧模型。

vLLM 的 PyTorch/CUDA wheel 具有版本耦合，可能替换 Kaggle 预装 torch。**先跑默认路径**；要试可选路径，请使用独立环境：

```bash
# Notebook 的 %%bash 单元执行；需 Internet 和足够磁盘。
python -m venv --system-site-packages /kaggle/working/grpo-vllm-env
/kaggle/working/grpo-vllm-env/bin/python -m pip install -r scripts/train_on_policy_script/requirements-grpo-vllm.txt
PATH=/kaggle/working/grpo-vllm-env/bin:$PATH \
  GRPO_OUTPUT_DIR=/kaggle/working/grpo-vllm-smoke \
  bash scripts/train_on_policy_script/run_kaggle_grpo.sh smoke --rollout-backend vllm
```

这条可选路径的安装、Triton 编译、HTTP API 和 NCCL 在本地均未做 GPU 集成验证，不保证 Kaggle 当前驱动可直接运行。软件栈不兼容时，默认双卡 Transformers 路径不受影响。
参考：[vLLM 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)、[支持架构](https://docs.vllm.ai/en/latest/models/supported_models/)、[0.19.1 Triton attention 支持 FP32 的源码](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/attention/backends/triton_attn.py)。

## 与作业参数对应

默认使用仓库 `cs336_alignment/prompts/r1_zero.prompt` 和现有 `r1_zero_reward_fn`，temperature=1、top-p=1、top-k=0。
`</answer>` 是默认停止字符串，并保留在回答中；EOS/pad 只在评分时从末尾移除，训练仍保留实际生成的 EOS token。
r1 reward 严格要求 `</think> <answer>` 中间有一个空格，模型不遵守格式时 reward=0。

| 参数 | smoke | experience | assignment / PDF |
|---|---:|---:|---:|
| 更新步数 | 2 | 50 | 200 |
| 每步题数 | 1 | 1 | 32 |
| group_size | 2 | 8 | 8 |
| 每步回答数 | 2 | 8 | 256 |
| 梯度累积次数 | 2 | 8 | 256（PDF 建议 32） |
| 生成上限 | 64 | 512 | 512 |
| 验证题数 | 2 | 32 | 1024 |
| 学习率 | 1e-5 | 1e-5 | 1e-5 |

assignment 仍用微批次 1，而不是 PDF 的微批次 8；保持每步回答数及 loss 缩放不变以控制显存。
训练从长度过滤后的集合按 seed 选最多 6400 道题；步内抽样不重复、跨步可重复。
默认 256-token prompt 上限会过滤极少量较长题，数量会打印；若要完整数据，增加上限并自行检查显存。
验证用过滤后 test 的前 N 条。正式调参建议从 train 留出验证集，保留 test 作最终评估。

```python
# 作业规模非常慢，先确认 experience 跑通。四个 seed 分开执行并保存各自输出。
!GRPO_OUTPUT_DIR=/kaggle/working/grpo-assignment-seed1 bash scripts/train_on_policy_script/run_kaggle_grpo.sh assignment --seed 1
```

梯度累积的语义是同一 rollout 批次内部拆微批次：
`microbatch = prompts_per_step * group_size / gradient_accumulation_steps`，必须整除。
同一题的 group_size 个回答连续排列，prompt/truth 同步重复。group_size >= 2，兼容你使用的样本标准差。
调温度/top-p/top-k 会改变采样分布，而现有 loss 不做修正，所以脚本会警告；严格 on-policy 请保留默认值。

脚本没有预跑 50 步，也没有证明 25% 准确率或四个 seed 的提升。基础模型 reward 可能接近 0；同组 reward 全相同则 advantage 全 0，loss/梯度可能为 0，不代表链路出错。

## 日志、保存、恢复

输出目录：

- `console.log`：wrapper 捕获 stdout/stderr，含 traceback。
- `metrics.jsonl`：train loss、裁剪前 grad norm、原策略 response-token entropy、total/format reward、回答 token 长度、步耗时；val accuracy、total/format reward、回答长度、题数。
- `samples.jsonl`：prompt、response、ground_truth、step 和 train/val 类型，`--log-samples` 控制每次条数。
- `config-from-step-*.json`：参数、模板文本、数据 SHA256。
- `checkpoint-NNNNNN/`：HF 权重/tokenizer 和 `trainer_state.pt`，包含 CPU Adam 状态、步数、数据采样器和 torch/CUDA RNG。

entropy 默认每步在更新前用已有 `get_reponse_log_probs(return_token_entropy=True)` 做额外无梯度前向，按所有回答的有效 token 求平均。有额外计算成本，`--entropy-every 10` 可降低记录频率。
GPU 显存指标只统计当前训练进程的 PyTorch allocator；vLLM 独立 server 占用需看 nvidia-smi，不要用训练进程的 GPU 1 指标估算 server 峰值。

checkpoint 体积很大：全量 FP32 权重约 5.53 GiB，Adam 状态约 11.06 GiB，合计约 **16.60 GiB**，还需 tokenizer/序列化开销。模型下载缓存另外占磁盘，运行前检查 `!df -h /kaggle/working`。wrapper 在 `/kaggle/temp` 存在且未设置 HF_HOME 时，将下载缓存放在那里；临时缓存不作为 Notebook 输出保留。脚本会在首次 rollout 前及每次保存前检查预计 checkpoint 空间，不足时明确报错。
默认 `--keep-checkpoints 1`：**先删除旧完整 checkpoint 再保存新 checkpoint**，控制磁盘峰值；保存期间故障可能失去上一份，因此重要实验请先备份到外部存储。
`--keep-checkpoints 0` 保留全部；`2` 保留两份，但峰值、长期配额需要相应增加。不会删除 `.incomplete` 半成品或覆盖同名 checkpoint。
日志一直追加，不受 checkpoint 清理影响。保存过程不完成时目录保持 `.incomplete`，不要用它恢复。

恢复必须重复同一个 profile 及额外训练参数，`--train-steps` 是包含已完成步数的总数：

```python
!bash scripts/train_on_policy_script/run_kaggle_grpo.sh smoke --resume /kaggle/working/grpo-smoke/checkpoint-000002 --train-steps 4
```

自定义模板/数据路径、学习率、采样配置都必须与原运行一致（文件路径可变，但内容指纹须相同）；可改总步数、日志/保存频率、输出目录。
恢复旧 checkpoint 时建议换输出目录，避免已经存在的同名 checkpoint。只加载自己信任的 trainer_state（使用 torch pickle）。
CPU 小模型已验证连续/恢复训练权重逐位一致；GPU 非确定性和跨版本数值差异仍可能影响轨迹。vLLM 每次请求按 seed、step 和批次索引设随机种子，未承诺跨硬件逐位一致。

可选 wandb：先 `%pip install wandb`，通过 Kaggle Secret 设置 `WANDB_API_KEY` 或自行登录，然后添加 `--wandb-project PROJECT --wandb-name NAME`；本地日志始终写入。

## 现有接口限制与验证范围

- `checkpoint.py:8–9` 固定 CUDA BF16/FlashAttention 2，本脚本自行加载 FP32/eager。
- `vllm_utils.py` 的 `start_server` 固定 BF16，新 backend 使用独立启动命令，继续调用原来的同步函数，不改原文件。
- `grpo_components.py:140–147` 内部 backward/step，metadata 为空。日志和裁剪通过显式 optimizer adapter/额外 entropy forward 补齐，未重写核心 loss。
- `utils.py:19–20` 分别 encode prompt/output，仅接受文本。脚本不自动加 BOS，保留生成 token 并检查 decode→encode 一致；若 byte token 解码有损会报错，彻底支持需要改原接口接收 token IDs。
- `utils.py:60` 不传 attention_mask；训练右 padding 在 OLMo 2 因果 attention 中不影响之前的有效 token，loss 已屏蔽 padding。生成则显式传左 padding mask。
- 仅支持零 dropout 模型（OLMo 2 默认满足），不开放尚未实现的 off-policy/其他 normalizer 变体。

本地检查：Python 3.12、torch 2.10.0+cu129、transformers 5.7.0，无 CUDA。使用真实小型随机 OLMo 2 + 本地 tokenizer 离线测试：

```bash
python -m pytest tests/test_grpo_onpolicy_script.py -q
```

覆盖已有 train step 的实际参数更新、梯度累积等价性、生成/训练/验证/保存/恢复、CPU AdamW 与标准 PyTorch AdamW 等价性、EOS 评分和 replica 同步、vLLM 启动配置。
另检查真实 OLMo tokenizer 的停止字符串和 GSM8K 读取。
**未验证：双 T4 峰值显存/吞吐/主机内存、Kaggle 安装、vLLM/NCCL GPU 集成、wandb 联网、真实 1B 收敛。**
