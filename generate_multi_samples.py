#!/usr/bin/env python3

"""Generate multiple virtual counselling sessions in parallel."""

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


THERAPIST_MODELS: List[str] = ["gpt-chat", "gpt-normal", "gpt-safety"]
CLIENT_MODEL = "default"
OUTPUT_DIR = Path("transcriptions_generated")
SCRIPT_DIR = Path(__file__).resolve().parent
PERSONAS_DIR = SCRIPT_DIR / "personas"
MAX_JOBS = int(os.getenv("MAX_JOBS", "4"))


@dataclass(frozen=True)
class Job:
    persona: str
    therapist_model: str

    @property
    def output_path(self) -> Path:
        return OUTPUT_DIR / f"{self.therapist_model}.{CLIENT_MODEL}.{self.persona}.log"

    def describe(self) -> str:
        return f"{self.therapist_model}/{self.persona}"


async def stream_process(job: Job) -> bool:
    if job.output_path.exists():
        print(f"[skip] {job.output_path}")
        return True

    print(f"[run ] therapist={job.therapist_model} client={CLIENT_MODEL} persona={job.persona}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(SCRIPT_DIR / "gen_virtual.py"),
        job.persona,
        "--model",
        job.therapist_model,
        "--client-model",
        CLIENT_MODEL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )

    success = True
    try:
        with job.output_path.open("w", encoding="utf-8") as log_file:
            assert process.stdout is not None
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                log_file.write(text)
                log_file.flush()
                sys.stdout.write(text)
                sys.stdout.flush()
        return_code = await process.wait()
        if return_code != 0:
            success = False
    finally:
        if not success:
            try:
                job.output_path.unlink(missing_ok=True)
            except OSError:
                pass

    if success:
        print(f"[done] {job.describe()}")
    else:
        print(f"[fail] {job.describe()}", file=sys.stderr)
    return success


async def run_jobs(jobs: List[Job]) -> bool:
    semaphore = asyncio.Semaphore(MAX_JOBS)
    results = []

    async def worker(job: Job) -> bool:
        async with semaphore:
            return await stream_process(job)

    for job in jobs:
        results.append(asyncio.create_task(worker(job)))

    success_flags = await asyncio.gather(*results, return_exceptions=False)
    return all(success_flags)


def build_jobs() -> List[Job]:
    if not PERSONAS_DIR.is_dir():
        raise SystemExit(f"No persona directory found at {PERSONAS_DIR}")

    persona_files = sorted(PERSONAS_DIR.glob("*.txt"))
    if not persona_files:
        raise SystemExit(f"No persona files found in {PERSONAS_DIR}")

    jobs = [
        Job(persona=persona_path.stem, therapist_model=model)
        for persona_path in persona_files
        for model in THERAPIST_MODELS
    ]
    return jobs


def main() -> int:
    jobs = build_jobs()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    success = loop.run_until_complete(run_jobs(jobs))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
