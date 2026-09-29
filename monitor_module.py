# monitor_module.py
import time
import psutil

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class ResourceMonitor:
    """
    with ResourceMonitor(audio_duration_sec) as m:
        result = model.transcribe(audio)
    print(m.report())
    """

    def __init__(self, audio_duration_sec: float = 0.0):
        self.audio_duration = audio_duration_sec
        self.ram_before  = 0.0
        self.vram_before = 0.0
        self.t_start     = 0.0
        self.latency_sec  = 0.0
        self.ram_used_mb  = 0.0
        self.vram_used_mb = 0.0
        self.rtf          = 0.0

    @staticmethod
    def _get_ram_mb():
        return psutil.virtual_memory().used / 1024 / 1024

    @staticmethod
    def _get_vram_mb():
        if not GPU_AVAILABLE:
            return 0.0
        gpus = GPUtil.getGPUs()
        return gpus[0].memoryUsed if gpus else 0.0

    def __enter__(self):
        self.ram_before  = self._get_ram_mb()
        self.vram_before = self._get_vram_mb()
        self.t_start     = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.latency_sec  = time.perf_counter() - self.t_start
        self.ram_used_mb  = max(0.0, self._get_ram_mb()  - self.ram_before)
        self.vram_used_mb = max(0.0, self._get_vram_mb() - self.vram_before)
        if self.audio_duration > 0:
            self.rtf = self.latency_sec / self.audio_duration

    def report(self):
        return {
            "latency_sec":  round(self.latency_sec,  4),
            "rtf":          round(self.rtf,           4),
            "ram_used_mb":  round(self.ram_used_mb,   2),
            "vram_used_mb": round(self.vram_used_mb,  2),
        }
