# 统一训练入口
# 用法: python train.py --config ../configs/warmup_stage.yaml
import sys
import os

# 获取项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.config import Config
from trainer.base_trainer import Trainer


# 在导入语句之后，读取配置之前添加此函数
def get_config_path(config_filename):
    """根据项目根目录获取配置文件的完整路径"""
    return os.path.join(project_root, 'configs', config_filename)

# 然后修改你的配置读取代码
if __name__ == '__main__':
    # 1. 读取配置
    cfg = Config.from_yaml(get_config_path("defaults.yaml"))
    # 实例化并运行
    trainer = Trainer(cfg)
    trainer.run()
