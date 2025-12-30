# YAML 读取与配置管理
# project/utils/config.py
import yaml
import os

class Config(dict):
    """
    配置类，支持字典访问和属性访问 (cfg.model.backbone)
    """
    def __init__(self, config_dict=None):
        super().__init__()
        if config_dict:
            for k, v in config_dict.items():
                self[k] = v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    @classmethod
    def from_yaml(cls, path):
        """从 yaml 文件加载配置"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls._dict_to_obj(data)

    @classmethod
    def _dict_to_obj(cls, data):
        """递归地将字典转换为 Config 对象"""
        if isinstance(data, dict):
            return cls({k: cls._dict_to_obj(v) for k, v in data.items()})
        elif isinstance(data, list):
            return [cls._dict_to_obj(i) for i in data]
        else:
            return data

    def merge_from_file(self, path):
        """合并另一个 yaml 文件的配置（用于覆盖默认配置）"""
        new_cfg = self.from_yaml(path)
        self.update(new_cfg)

# ================= 测试代码 =================
if __name__ == "__main__":
    # 假设你已经在 configs 目录下创建了 defaults.yaml
    cfg_path = "../configs/defaults.yaml"
    
    # 尝试加载
    if os.path.exists(cfg_path):
        cfg = Config.from_yaml(cfg_path)
        print("加载成功!")
        print(f"Backbone: {cfg.model.vision.backbone}")
        print(f"Learning Rate: {cfg.train.lr}")
        print(f"Tasks: {cfg.model.class_head.task_dicts}")

        for item in cfg.model.class_head.task_dicts:
            # item 是一个只有一个键值对的字典，例如 {'road': 8}
            # 为了兼容性，我们获取第一个键和值
            if isinstance(item, dict):
                key = list(item.keys())[0]
                val = list(item.values())[0]
                print(key,val)
    else:
        print(f"请先创建 {cfg_path} 文件")