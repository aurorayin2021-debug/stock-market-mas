"""在指定时间窗口监督一次批次；不保存密钥、不自动重试。"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _time(value):
    result = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("时间必须包含时区。")
    return result


def _now():
    return datetime.now(timezone.utc)



@contextmanager
def _lock(path):
    """进程级非阻塞锁；退出或崩溃后由系统释放。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("已有定时批次正在运行。") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)

def supervise(project, prediction_date, start_at, stop_at, hard_stop, *, concurrency=20, model="deepseek-flash"):
    start, stop, hard = _time(start_at), _time(stop_at), _time(hard_stop)
    day = datetime.strptime(prediction_date, "%Y-%m-%d").date().isoformat()
    if not start < stop < hard:
        raise ValueError("开始时间、停止提交时间和强制结束时间必须依次递增。")
    if not start <= _now() < stop:
        raise ValueError("当前不在允许运行的时间窗口内。")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("并发数必须为正整数。")
    root = Path(project).resolve()
    script = root / "run_analysts_batch.py"
    if not script.is_file():
        raise FileNotFoundError("找不到批处理脚本。")
    logs = root / "outputs" / "scheduled_runs"
    logs.mkdir(parents=True, exist_ok=True)
    name = "batch-" + _now().strftime("%Y%m%dT%H%M%SZ")
    log = logs / (name + ".log")
    report = logs / (name + ".json")
    args = [sys.executable, "-X", "utf8", "-B", str(script), "--all", "--date", day, "--roles", "news", "kline",
            "--concurrency", str(concurrency), "--model", model,
            "--start-at", start.isoformat(), "--stop-at", stop.isoformat()]
    # 再次检查，避免准备日志过程中错过截止时间。
    if not start <= _now() < stop:
        raise ValueError("准备完成时已经超过停止提交时间。")
    terminated = False
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    audit_file = logs / (name + ".audit.json")
    audit = root / "audit_analysts.py"
    with _lock(logs / ".batch.lock"):
        if not start <= _now() < stop:
            raise ValueError("取得运行锁时已经超过停止提交时间。")
        with log.open("x", encoding="utf-8") as stream:
            proc = subprocess.Popen(args, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=flags)
            try:
                while proc.poll() is None:
                    left = (hard - _now()).total_seconds()
                    if left <= 0:
                        proc.terminate()
                        proc.wait()
                        terminated = True
                        break
                    time.sleep(min(1, left))
            except BaseException:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                raise
        audit_status = "missing"
        if audit.is_file():
            try:
                with audit_file.open("x", encoding="utf-8") as stream:
                    audit_run = subprocess.run([sys.executable, "-X", "utf8", "-B", str(audit), "--date", day],
                        cwd=root, stdout=stream, stderr=subprocess.STDOUT, timeout=120, creationflags=flags)
                audit_status = "ok" if audit_run.returncode == 0 else "failed"
            except (OSError, subprocess.TimeoutExpired):
                audit_status = "failed"
        result = {"prediction_date": day, "exit_code": proc.returncode, "hard_stopped": terminated,
                  "log": str(log), "audit_status": audit_status,
                  "audit_file": str(audit_file) if audit_file.exists() else None}
        with report.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    result["report"] = str(report)
    return result



def main():
    p = argparse.ArgumentParser(description="在指定时段运行并检查一次新闻与 K 线批次。")
    p.add_argument("--project", default=str(Path(__file__).resolve().parent))
    p.add_argument("--date", required=True)
    p.add_argument("--start-at", required=True)
    p.add_argument("--stop-at", required=True)
    p.add_argument("--hard-stop", required=True)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--model", default="deepseek-flash")
    a = p.parse_args()
    result = supervise(a.project, a.date, a.start_at, a.stop_at, a.hard_stop, concurrency=a.concurrency, model=a.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["exit_code"] != 0 or result["hard_stopped"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
