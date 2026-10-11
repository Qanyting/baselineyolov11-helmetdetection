class DetectionError(Exception):
    """可以安全地显示给用户的检测失败原因。"""


class ModelUnavailable(DetectionError):
    """权重缺失、依赖缺失或模型加载失败。"""
