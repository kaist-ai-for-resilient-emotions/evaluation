import os
from pathlib import Path
from typing import Mapping, Optional


def load_env(env_path: Optional[Path] = None, *, override: bool = False) -> Mapping[str, str]:
    """간단한 .env 로더.

    Args:
        env_path: 읽을 .env 파일 경로 (기본값: 현재 파일과 동일한 디렉터리의 .env).
        override: True면 이미 설정된 환경 변수도 덮어씀.

    Returns:
        로드된 키-값 쌍을 담은 dict.
    """
    if env_path is None:
        env_path = Path(__file__).parent / ".env"

    if not env_path.is_file():
        return {}

    loaded = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value

    return loaded
