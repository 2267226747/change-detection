import logging
import os
import sys
from datetime import datetime

from contextlib import contextmanager

@contextmanager
def suppress_console_logging(logger):
    """
    临时移除 logger 中的所有 StreamHandler（控制台输出），退出后恢复。
    适用于：仅将日志写入文件，不打印到终端。
    """
    # 找出所有 StreamHandler（排除 FileHandler）
    stream_handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    
    # 临时移除
    for h in stream_handlers:
        logger.removeHandler(h)
    
    try:
        yield logger
    finally:
        # 恢复
        for h in stream_handlers:
            logger.addHandler(h)


def setup_logger(save_dir, distributed_rank=0, filename="log/train.log", name="SVI_Project"):
    """
    初始化 Logger，自动添加时间戳，并处理分布式进程的日志级别。

    Args:
        save_dir: 日志保存的根目录 (例如 ./results/exp1/)
        distributed_rank: 进程 ID。Rank 0 记录 INFO，其他 Rank 只记录 WARNING。
        filename: 日志文件的相对路径 (例如 "log/train.log")
        name: Logger 的名称
    Returns:
        logger 对象
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # 防止向上传播导致重复打印

    # 1. 设置日志级别
    # 主进程记录 INFO，子进程只记录 WARNING (防止报错时子进程静默)
    logger.setLevel(logging.INFO if distributed_rank == 0 else logging.WARNING)

    # 2. 避免重复添加 Handler
    if logger.handlers:
        return logger

    # 3. 定义通用格式
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 4. Console Handler (所有进程都需要，以便在终端看到报错)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 5. File Handler (仅主进程 Rank 0)
    if distributed_rank == 0 and save_dir:
        # --- 时间戳处理 ---
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # 分离路径中的 目录部分 和 文件名部分
        # 假设 filename="log/train.log" -> rel_dir="log", base_name="train.log"
        rel_dir = os.path.dirname(filename)
        base_name = os.path.basename(filename)

        # 分离文件名和后缀 -> "train", ".log"
        name_root, ext = os.path.splitext(base_name)

        # 组合新文件名 -> "train_2023-10-27_15-30.log"
        new_filename = f"{name_root}_{current_time}{ext}"

        # 组合最终的绝对目录路径 -> ./results/exp1/log/
        final_log_dir = os.path.join(save_dir, rel_dir)
        os.makedirs(final_log_dir, exist_ok=True)

        # 组合最终文件路径
        final_log_path = os.path.join(final_log_dir, new_filename)

        # 创建 FileHandler (使用 'w' 模式，因为文件名是唯一的)
        fh = logging.FileHandler(final_log_path, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # 在控制台打印一下日志文件的实际位置，方便查找
        print(f"Logging to: {final_log_path}")

    return logger