import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from env.hardware.hardware_cache import HardwareCache
from env.hardware.hardware_config import HardwareConfig
c = HardwareCache(HardwareConfig(force_cpu_mode=True))
print("hasattr:", hasattr(c, "chunk_in_cache"))
