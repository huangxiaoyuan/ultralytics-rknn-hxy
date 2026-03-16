##账号dglg
#密码DZ@8b119

#节点
#python -m torch.distributed.launch --nproc_per_node=2 --node_rank=0 --nnodes=1 --master_addr=localhost --master_port=1234 train_detect_server.py --batch 32 --device 0,1
# nohup python -m torch.distributed.launch --node_rank 0 --nproc_per_node=2  train_detect_server.py  --batch 32 --device 0,1 &
#torch.distributed.launch已弃用，使用

# 新建
screen -S train
torchrun --node_rank 0 --nproc_per_node=2 --nnodes=1 --rdzv_endpoint=localhost:1234 train_detect_server.py --batch 32
# 断开
Ctrl+A → D
# 重新连接
screen -r train


#训练指令
# torchrun --node_rank 0 --nproc_per_node=2 --nnodes=1 --rdzv_endpoint=localhost:1234 train_detect_server.py --batch 32 
# 使用nohup切换终端，后台运行
# nohup torchrun --node_rank 0 --nproc_per_node=2 --nnodes=1 --rdzv_endpoint=localhost:1234 train_detect_server.py --batch 32 &
# tail -f nohup.out




# 新建一个命名的 session
tmux new -s train
# 在里面启动训练
python train_detect_server.py
# 关闭终端 / 断开连接（训练继续在后台跑）
Ctrl+B，然后按 D
# 下次重新连接查看训练进度
tmux attach -t train



#多节点
# 主节点（Node 0）
#torchrun --nproc_per_node=2 --nnodes=2 --rdzv_endpoint=192.168.1.100:29500 train_detect_server.py ...
# 从节点（Node 1）
#torchrun --nproc_per_node=2 --nnodes=2 --rdzv_endpoint=192.168.1.100:29500 train_detect_server.py ...

#screen -S your_session_name
#screen -r your_session_name
#创建新窗口：Ctrl+b + c
#切换窗口：Ctrl+b + n/p

# python -m torch.distributed.launch --node_rank 2,3 --nproc_per_node=2  train_detect_server.py  --batch 32 --device 0,1




**PyTorch 分布式训练（DDP, Distributed Data Parallel）** 的标准启动方式。下面是每个参数的详细意义：
### 1. `torchrun` 本体
它是 PyTorch 提供的启动器（旧版本叫 `torch.distributed.launch`）。它的作用是：
*   **自动管理进程**：它会帮你启动多个 Python 进程（每个 GPU 一个）。
*   **故障容错**：如果某个进程挂了，`torchrun` 会尝试重启所有进程。
*   **环境变量注入**：它会自动给你的脚本注入 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 等环境变量，供代码内部调用。

### 2. 基础设施参数（Infrastructure Parameters）

*   **`--nnodes=1`**
    *   **含义**：参与训练的总“节点”数（通常一台机器就是一个节点）。
    *   **场景**：如果你只在这一台电脑上练，就设为 1；如果是多台机器联网练，就设为机器的总数。

*   **`--node_rank 0`**
    *   **含义**：当前这台机器在所有节点中的编号（从 0 开始）。
    *   **场景**：单机训练时固定为 0。如果是多机训练，第一台机器是 0，第二台是 1，以此类推。

*   **`--nproc_per_node=2`**
    *   **含义**：在**当前这台机器**上启动多少个进程。
    *   **建议**：通常**等于该机器上的 GPU 数量**。你设为 2，意味着 `torchrun` 会启动 2 个进程，分别对应你的 GPU 0 和 GPU 1。

*   **`--rdzv_endpoint=localhost:1234`**
    *   **含义**：**Rendezvous（汇合点）地址**。
    *   **作用**：分布式训练时，各个进程需要互相知道对方的 IP 和端口。`localhost:1234` 表示在本地启动一个服务，端口是 1234，所有进程都去这里“报到”。
    *   **注意**：如果你在一台机器上运行，写 `localhost:端口` 即可；端口号可以随便改（1024-65535 之间）。

### 3. 你的训练脚本及参数

*   **`train_detect_server.py`**
    *   这是你要执行的 Python 脚本。`torchrun` 实际上是在后台执行了两次类似 `python train_detect_server.py` 的操作。

*   **`--batch 32`**
    *   这是传递给 `train_detect_server.py` 的**自定义参数**。
    *   **重要提示**：在 DDP 模式下，这个 `32` 通常指的**是“单卡（单进程）的 Batch Size”**。因为你开了 2 个进程（`nproc_per_node=2`），所以**全局总 Batch Size 是 32 × 2 = 64**。

### 进阶提示：
如果在代码中想要获取当前是哪张卡，通常在 Python 脚本里这样写：
```python
import os
local_rank = int(os.environ["LOCAL_RANK"]) # torchrun 会自动注入这个环境变量
torch.cuda.set_device(local_rank)
```