# Kaggle 双 T4：GRPO on-policy experiment

训练入口：`scripts/train_on_policy_script/train_on_policy_script.py`。

阅读作业 PDF §4.2–4.3 后修订的版本。只修改/新增 `scripts/` 和对应测试，不修改你的 `cs336_alignment` 实现；没有 monkey patch。已新增独立混合精度 train step，复用你的 GRPO loss 组件；原来的 train step 文件保持不变。

## 直接在 Kaggle 运行

Notebook 选择 **GPU T4 x2**，打开 Internet。在 Notebook 中执行（仓库地址替换为你自己的）：

```python
!git clone <你的仓库地址> /kaggle/working/assignment5-alignment
%cd /kaggle/working/assignment5-alignment
# 已经 clone 时改用 !git pull
%pip install --no-cache-dir -r scripts/train_on_policy_script/requirements-grpo-vllm.txt
```

不要安装项目 `[gpu]` extra：其中含 FlashAttention/vLLM 的其他依赖，且项目 Python 版本限制不一定匹配 Kaggle。
当前默认是 vLLM 推理，必须安装上面的 vLLM 依赖。固定 vLLM 0.19.1，依赖 torch/torchaudio 2.10.0 和 torchvision 0.25.0，pip 可能替换 Kaggle 预装包；使用 `--no-cache-dir` 避免额外缓存 wheel。安装完成后重启 kernel，再进入仓库运行。若只用 Transformers，则安装 `requirements-grpo.txt` 并显式指定 `--rollout-backend transformers`。
本脚本直接读取仓库现有 `data/gsm8k/train.jsonl` 和 `test.jsonl`。

先检查 GPU，再执行两步小规模验证：

```python
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.version.cuda); print([(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())])"
!bash scripts/train_on_policy_script/run_kaggle_grpo.sh smoke
```

默认输出位置：`/kaggle/working/grpo-smoke-amp/`。应看到训练/验证指标、样例和 `checkpoint-000002/`。
两步检查使用 64 个生成 tokens，可能截断解题过程，只验证执行链路，不用于判断训练效果。

通过后运行 50 步体验：

```python
!bash scripts/train_on_policy_script/run_kaggle_grpo.sh experience
```

默认输出：`/kaggle/working/grpo-experience-amp/`。
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

## 双 GPU 与混合精度

本版默认启用 FP16 混合精度（`--precision fp16`），没有 BF16。默认输出目录改为 `grpo-<profile>-amp`，与之前 FP32 运行的 checkpoint 分开，避免同名冲突。
一个 Python 进程，不用 DDP/FSDP 或 torchrun：

- GPU 0：FP16 policy 参数与梯度；FP16 autocast 前向，梯度检查点，默认 SDPA attention（由 PyTorch 自动选择支持的 kernel，不要求 FlashAttention 2）。
- CPU：FP32 主权重和 PyTorch AdamW 状态，betas=(0.9,0.95)，weight_decay=0。
- GPU 1：默认 FP16 vLLM server，连续批处理，默认最多 32 个活动序列，开启 prefix caching。每批采样及更新后验证前都同步当前 policy 并清理旧 cache。可显式指定 `--rollout-backend transformers` 作为替代。
- log-softmax、advantage、loss 和 entropy 使用 FP32；rotary 等浮点 buffer 保持 FP32。

新增 `amp_train_step.py`：按整批 rollout 计算 group advantage；所有微批次使用同一个 loss scale，`scaler.scale(loss).backward()` 累积梯度。之后把 FP16 模型梯度转成 **CPU FP32 主权重的梯度**，再 `scaler.unscale_(optimizer)`，记录范数、裁剪、`scaler.step()`、`scaler.update()`，最后把更新后的主权重复制回 GPU 模型。
GradScaler 操作的是 FP32 master gradients，避免对 FP16 parameter gradients 直接反缩放时报错。梯度 inf/NaN 会丢弃本批更新、降低 loss scale，并在下一步用当前模型重新采样，不复用 rollout。初始 scale 默认 128，可用 `--amp-init-scale` 修改。
如果前向 loss 已经非有限，会明确报错停止；loss scaling 不能修复 FP16 前向激活越界。

这套顺序遵循 [PyTorch AMP 的累积、反缩放与裁剪规则](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html)。没有将 FP16 参数交给普通 AdamW 直接更新：AdamW 更新的是 CPU 上的 FP32 master weights。
每批仍最多一次参数更新，loss 复用原有 `compute_policy_gradient_loss` / `aggregate_loss_across_microbatch` 等组件。

真实 OLMo 配置统计为 1,484,916,736 个参数，以下都是下界或估算，不是 T4 实测峰值：

| 位置 | 默认 FP16 配置 |
|---|---:|
| GPU 0 参数 + 梯度 | 约 **5.53 GiB**，另加激活、FP32 logits/log-softmax、工作区 |
| GPU 1 参数 | 约 **2.77 GiB**，另加 KV cache/工作区 |
| CPU 主权重 + Adam 状态 | 约 **16.60 GiB** |
| CPU 更新阶段的完整 FP32 梯度 | 另外约 **5.53 GiB**，更新后释放 |
| 模型 checkpoint | 权重约 **2.77 GiB**，不含主权重副本和优化器状态 |

CPU 更新阶段合计约 22.13 GiB 加临时张量/加载开销，需要足够主机 RAM；建议使用约 30 GiB 的主机内存环境。CPU AdamW 与 PCIe 传输仍有成本，不承诺具体加速倍数。
微批次只前向到自身最长有效序列，减少无效 padding 计算；entropy 在这次前向里按块计算，不再额外前向一遍模型。
`--precision fp32` 可用于比较/排查，原有 FP32 train step 作为参考实现保留。CPU smoke test 自动使用 FP32；离线单测另外通过 CPU FP16 验证了缩放/溢出处理，但不能替代 CUDA kernel 检查。

单卡调试需指定 `--rollout-backend transformers --rollout-gpu -1 --rollout-batch-size 1`，默认 vLLM 路径需要独立 GPU。替代模型需支持所选 attention；必要时加 `--attn-implementation eager`。

## T4 能不能使用 vLLM？

**硬件支持，但具体 Kaggle 软件栈需要实测。** 官方安装页列出 T4/compute capability 7.5；OLMo 2 在支持的架构列表中。
T4 不支持本方案需要的原生 BF16；不能沿用仓库 `vllm_utils.start_server()` 固定的 `--dtype bfloat16`。

用户后续要求加速推理，当前 CUDA 默认使用 `--rollout-backend vllm`；原先不使用 vLLM 的约束已由此次要求更新，Transformers 仍可显式选择：

- 固定 vLLM 0.19.1，独立子进程运行 GPU 1；`float16`（FP32 模式则为 `float32`）、`TRITON_ATTN`、`--enforce-eager`，不强制 FlashAttention 2。
- 与训练 policy 参数 dtype 一致；采样和训练均采用相同 FP16 权重，归一化仍为 FP32。
- 复用仓库 `init_weight_sync` / `sync_policy_weights` 的 NCCL 权重传输和缓存清理，不使用过期权重。
- 传入 token prompts，显式关闭额外 BOS；每道题只传一次，`n=group_size` 并行采样，由 vLLM 为不同 child 分配不同 seed。按 index 恢复题目/回答顺序，保留实际 token IDs。验证为 greedy、n=1。
- `--rollout-batch-size 32` 同时设置 server 的 max-num-seqs，按 group 大小分配请求，减少 HTTP 调用并复用同题 prompt prefill；它不是每次参数更新的总 rollout 数，训练批次仍由 prompts-per-step × group-size 决定。
- server 日志写入 `vllm-server.log`，启动失败不会悄悄回退至旧模型。

vLLM 的 PyTorch/CUDA wheel 具有版本耦合，不能只修改运行参数而不安装依赖。上面的 Notebook 安装方式可直接用；若希望隔离预装环境，也可使用独立环境（环境目录也需要磁盘，优先使用 /kaggle/temp）：

```bash
# Notebook 的 %%bash 单元执行；需 Internet 和足够磁盘。
python -m venv --system-site-packages /kaggle/temp/grpo-vllm-env
/kaggle/temp/grpo-vllm-env/bin/python -m pip install --no-cache-dir -r scripts/train_on_policy_script/requirements-grpo-vllm.txt
PATH=/kaggle/temp/grpo-vllm-env/bin:$PATH \
  GRPO_OUTPUT_DIR=/kaggle/working/grpo-vllm-smoke \
  bash scripts/train_on_policy_script/run_kaggle_grpo.sh smoke --rollout-backend vllm
```

本地没有安装 vLLM，也没有 CUDA；没有实测此组合的安装、Triton 编译和 NCCL GPU 集成，不能承诺 Kaggle 当前驱动直接兼容。启动时会先检查依赖版本，缺少时在加载模型前报错。首次启动需要加载权重、编译 kernel，可能比后续运行慢；可查看输出目录的 `vllm-server.log` 了解进度。server 出错也会指向该文件，不会悄悄回退到 Transformers。若需要临时切换，用 `--rollout-backend transformers --rollout-batch-size 4`。
并行 child seed 规则已核对 [vLLM 0.19.1 源码](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/engine/parallel_sampling.py)。参考：[vLLM 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)、[支持架构](https://docs.vllm.ai/en/latest/models/supported_models/)、[0.19.1 Triton attention 支持 FP32 的源码](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/attention/backends/triton_attn.py)。

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

- `console.log`：wrapper 捕获 stdout/stderr，含 traceback。终端会实时打印阶段进度、每批生成完成数量，以及每步 loss/reward/grad norm 等指标。
- `progress.log`：训练脚本直接记录同步、生成、entropy、更新、验证和保存阶段，即使不通过 wrapper 启动也会写入。
- `metrics.jsonl`：train loss、裁剪前 grad norm、原策略 response-token entropy、total/format reward、回答 token 长度、步耗时，以及 `train/amp_scale` / `train/optimizer_step_skipped`；val accuracy、total/format reward、回答长度、题数。溢出批次的 grad_norm 记为 null，避免写入无效 JSON。
- `samples.jsonl`：prompt、response、ground_truth、step 和 train/val 类型，`--log-samples` 控制每次条数。
- `config-from-step-*.json`：参数、模板文本、数据 SHA256。
- `checkpoint-NNNNNN/`：HF 权重/tokenizer 和很小的 `trainer_state.pt`，只保存步数、配置、数据采样器和 torch/CUDA RNG，**不保存优化器状态**。

entropy 默认每步复用训练前向的 FP32 log-prob，以小块无梯度运算按回答有效 token 求平均，不再增加模型前向次数；`--entropy-every 10` 可降低统计频率。它记录的是采样后、更新前的原始策略 entropy。
GPU 显存指标只统计当前训练进程的 PyTorch allocator；vLLM 独立 server 占用需看 nvidia-smi，不要用训练进程的 GPU 1 指标估算 server 峰值。

checkpoint 只保存模型、tokenizer 和少量训练元数据，**不保存 Adam 状态、FP32 master weights 或 GradScaler 状态**。默认 FP16 模型权重约 **2.77 GiB**，空间检查约为 **2.89 GiB**；`--precision fp32` 的权重仍约 5.53 GiB。模型下载缓存另外占磁盘，运行前检查 `!df -h /kaggle/working`。wrapper 在 `/kaggle/temp` 存在且未设置 HF_HOME 时，将下载缓存放在那里；临时缓存不作为 Notebook 输出保留。脚本会在首次 rollout 前及每次保存前检查预计 checkpoint 空间，不足时明确报错。
默认 `--keep-checkpoints 1`：**先删除旧完整 checkpoint 再保存新 checkpoint**，控制磁盘峰值；保存期间故障可能失去上一份，因此重要实验请先备份到外部存储。
`--keep-checkpoints 0` 保留全部；`2` 保留两份，但峰值、长期配额需要相应增加。不会删除 `.incomplete` 半成品或覆盖同名 checkpoint。
日志一直追加，不受 checkpoint 清理影响。保存过程不完成时目录保持 `.incomplete`，不要用它恢复。

恢复必须重复同一个 profile 及额外训练参数，`--train-steps` 是包含已完成步数的总数：

```python
!bash scripts/train_on_policy_script/run_kaggle_grpo.sh smoke --resume /kaggle/working/grpo-smoke-amp/checkpoint-000002 --train-steps 4
```

自定义模板/数据路径、学习率、采样配置都必须与原运行一致（文件路径可变，但内容指纹须相同）；可改总步数、日志/保存频率、输出目录。
恢复旧 checkpoint 时建议换输出目录，避免已经存在的同名 checkpoint。只加载自己信任的 trainer_state（使用 torch pickle）。
**恢复时加载模型、步数和随机数状态，但 FP32 主权重从保存的模型初始化，Adam 动量与步数、AMP loss scale 重新初始化，终端会明确提示；因此恢复后的轨迹不与不中断训练逐位一致。** 推理 backend、批量大小、精度和 attention 配置允许变更，其余训练参数仍检查一致性。 CPU 小模型已验证轻量 checkpoint 可加载并继续训练。vLLM 每次请求按 seed、step 和批次索引设随机种子，未承诺跨硬件逐位一致。

可选 wandb：先 `%pip install wandb`，通过 Kaggle Secret 设置 `WANDB_API_KEY` 或自行登录，然后添加 `--wandb-project PROJECT --wandb-name NAME`；本地日志始终写入。

## 现有接口限制与验证范围

- `checkpoint.py:8–9` 固定 CUDA BF16/FlashAttention 2，本脚本独立加载并配置 FP16/SDPA 或 FP32。
- `vllm_utils.py` 的 `start_server` 固定 BF16，新 backend 使用独立启动命令，继续调用原来的同步函数，不改原文件。
- `grpo_components.py:140–147` 内部直接 backward/step，不能按标准 AMP 顺序接入。用户授权后，新增 `amp_train_step.py` 处理缩放、累积、反缩放、裁剪和更新；原 train step 不改，核心 loss 组件仍复用原实现。
- `utils.py:19–20` 分别 encode prompt/output。生成 token 解码再编码不保证相同，例如真实 OLMo tokenizer 的 `[71,4896]` 解码为 `hello`，重新编码得到 `[15339]`。脚本用 `TokenizedResponse`（携带原始 token IDs 的字符串）和显式 `RolloutTokenizer` 适配器，把实际采样的 IDs 交给原有训练/entropy 函数；prompt 仍按正常方式编码。评分和日志使用可读文本，同样文本对应的不同 token 序列也分别保留，无需修改原函数或 monkey patch。验证不再因正常的非唯一 token 分词而中断。
- 原 `utils.py:60` 不传 attention_mask。新 step 按真实 token 长度构建 mask，避免 pad==EOS 时误屏蔽真实 EOS，并裁掉微批次末尾无效 padding。生成仍显式传左 padding mask。
- 仅支持零 dropout 模型（OLMo 2 默认满足），不开放尚未实现的 off-policy/其他 normalizer 变体。

本地检查：Python 3.12、torch 2.10.0+cu129、transformers 5.7.0，无 CUDA。使用真实小型随机 OLMo 2 + 本地 tokenizer 离线测试：

```bash
python -m pytest tests/test_grpo_onpolicy_script.py tests/test_grpo_amp_step.py tests/test_grpo_vllm_requests.py -q
```

覆盖已有 train step 的实际参数更新、梯度累积等价性、生成/训练/验证/保存/恢复、CPU AdamW 与标准 PyTorch AdamW 等价性、EOS 评分和 replica 同步、vLLM 启动配置；另有不满足 decode→encode 一致性的回归测试，确认训练标签、mask、长度与原始采样 token 一致。
新增混合精度测试覆盖 FP32 与原 step 的更新等价性、CPU FP16 + GradScaler 累积、FP32 Adam 主权重更新、强制梯度溢出跳过及后续恢复，以及 FP16 checkpoint 确实不含优化器。另检查真实 OLMo tokenizer 的停止字符串和 GSM8K 读取。
测试还包括 CUDA GradScaler + CPU master gradients 的正常更新/溢出用例；本地没有 CUDA，因此这两项跳过。在 Kaggle 安装 pytest 后运行上面的测试命令会实际执行。
**未验证：本次混合精度版本的双 T4 峰值显存/吞吐/主机内存、CUDA scaler 到 CPU 梯度的集成、vLLM/NCCL GPU 集成、wandb 联网、真实 1B 收敛。** 之前用户提供的 FP32 smoke 结果不代表新版本性能。

默认日志开头应出现 `Rollout backend=vllm, concurrency=32`，随后为 vLLM server 启动/权重同步及 `[vllm/generate/train]` 进度。HTTP payload、n 分组、返回索引和 token IDs 均有不依赖 vLLM 安装的离线测试；这些协议检查不等同于真实推理引擎测试。
