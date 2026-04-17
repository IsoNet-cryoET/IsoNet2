import os
import tempfile
from pathlib import Path


def _configure_pytorch_kernel_cache():
	if os.environ.get("USE_PYTORCH_KERNEL_CACHE") == "0":
		return

	configured_path = os.environ.get("PYTORCH_KERNEL_CACHE_PATH")
	cache_candidates = []

	if configured_path:
		cache_candidates.append(Path(configured_path).expanduser())
	else:
		xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
		if xdg_cache_home:
			cache_candidates.append(Path(xdg_cache_home).expanduser() / "torch" / "kernels")
		else:
			cache_candidates.append(Path.home() / ".cache" / "torch" / "kernels")

	user_id = getattr(os, "getuid", lambda: "user")()
	cache_candidates.append(Path(tempfile.gettempdir()) / f"pytorch-kernels-{user_id}")

	for cache_dir in cache_candidates:
		try:
			cache_dir.mkdir(parents=True, exist_ok=True)
			test_file = cache_dir / ".write_test"
			with open(test_file, "a", encoding="utf-8"):
				pass
			test_file.unlink()
			os.environ["PYTORCH_KERNEL_CACHE_PATH"] = str(cache_dir)
			return
		except OSError:
			continue


_configure_pytorch_kernel_cache()
