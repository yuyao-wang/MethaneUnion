# resize all tif images to 224x224 (skip already processed but keep full csv)
import os
from pathlib import Path
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import time

import numpy as np
import pandas as pd
import tifffile as tiff

TRAIN_CSV = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16/train.csv"
TEST_CSV  = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16/test.csv"

OLD_ROOT = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16"
NEW_ROOT = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/s2_90360_temporal_CDSE0_gee90360_2024_16_224"

IMG_COLS = ["image_path", "s2_pre_path", "s2_pre_pre_path"]

OUT_H = 224
OUT_W = 224

def make_new_path(old_path: str) -> str:
    rel = Path(old_path).relative_to(OLD_ROOT)
    return str(Path(NEW_ROOT) / rel)

def resize_bilinear_chw(img: np.ndarray, out_h=224, out_w=224) -> np.ndarray:
    # img: C x H x W
    if img.ndim != 3:
        raise ValueError(f"Expected 3D CHW, got {img.shape}")

    c, h, w = img.shape
    if h == out_h and w == out_w:
        return img  # already resized

    y = np.linspace(0, h - 1, out_h, dtype=np.float32)
    x = np.linspace(0, w - 1, out_w, dtype=np.float32)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = (x - x0.astype(np.float32))[None, :]   # 1 x W
    wy = (y - y0.astype(np.float32))[:, None]   # H x 1

    Ia = img[:, y0[:, None], x0[None, :]]  # C x H x W
    Ib = img[:, y0[:, None], x1[None, :]]
    Ic = img[:, y1[:, None], x0[None, :]]
    Id = img[:, y1[:, None], x1[None, :]]

    out = (Ia * (1 - wx) * (1 - wy) +
           Ib * wx * (1 - wy) +
           Ic * (1 - wx) * wy +
           Id * wx * wy)

    return out.astype(img.dtype, copy=False)

def ensure_chw(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"Unexpected ndim={img.ndim}, shape={img.shape}")

    # CHW
    if img.shape[0] in (1, 12) and img.shape[1] >= 1 and img.shape[2] >= 1:
        return img
    # HWC -> CHW
    if img.shape[-1] in (1, 12):
        return np.transpose(img, (2, 0, 1))

    raise ValueError(f"Cannot infer CHW/HWC from shape={img.shape}")

def is_done(dst_path: str, out_h=224, out_w=224) -> bool:
    """
    ✅ 快速判断：文件存在 + 只读 TIFF header 拿 shape（不解码整张图）
    避免 tiff.imread 在网络盘上解码/随机 IO 导致尾部卡住。
    """
    p = Path(dst_path)
    if not p.exists():
        return False
    try:
        with tiff.TiffFile(str(p)) as tf:
            shape = tf.series[0].shape  # e.g. (C,H,W) or (H,W,C)
        if len(shape) != 3:
            return False

        # CHW
        if shape[0] in (1, 12) and shape[1] == out_h and shape[2] == out_w:
            return True
        # HWC
        if shape[-1] in (1, 12) and shape[0] == out_h and shape[1] == out_w:
            return True

        return False
    except Exception:
        return False

def resize_tif(src_path: str, dst_path: str):
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ 防止“半写入文件”导致以后检查/解码卡住：写到 tmp 再原子替换
    tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")

    img = tiff.imread(src_path)
    img = ensure_chw(img)
    out = resize_bilinear_chw(img, OUT_H, OUT_W)

    tiff.imwrite(str(tmp), out)
    tmp.replace(dst_path)

def safe_resize_one(src_path: str, dst_path: str, skip_if_done=True):
    """
    返回 (src_path, status, info)
    status: "ok" | "skip" | "fail"
    """
    try:
        if skip_if_done and is_done(dst_path, OUT_H, OUT_W):
            return (src_path, "skip", "")
        resize_tif(src_path, dst_path)
        return (src_path, "ok", "")
    except Exception as e:
        return (src_path, "fail", f"{repr(e)}\n{traceback.format_exc()}")

def run_parallel(
    paths,
    new_paths,
    workers=16,
    skip_if_done=True,
    per_task_timeout_sec=120,     # 单任务“软超时”
    stall_report_sec=120,         # 每隔多久打印一次 pending 示例
    timeout_list_path=None,       # 例如: "/tmp/timeouts_train.txt"
):
    """
    软超时策略：
    - 无法强杀卡住 I/O 的线程
    - 但主线程在超过 per_task_timeout_sec 后“放弃等待”该任务
    - 记录 timeout 的 src_path，继续收尾，避免永远 pending
    """
    bad = 0
    skipped = 0
    finished = 0
    total = len(paths)

    # 每个 future 的起始时间
    fut_start = {}
    fut2src = {}

    timeouts = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending = set()

        for src, dst in zip(paths, new_paths):
            fut = ex.submit(safe_resize_one, src, dst, skip_if_done)
            pending.add(fut)
            fut2src[fut] = src
            fut_start[fut] = time.time()

        last_stall_report = time.time()

        while pending:
            done_set, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)

            # 处理已完成
            for fut in done_set:
                try:
                    src, status, info = fut.result()
                except Exception as e:
                    # 极少数情况下 fut.result() 自身异常
                    src = fut2src.get(fut, "<unknown>")
                    status = "fail"
                    info = repr(e)

                finished += 1

                if status == "fail":
                    bad += 1
                    print(f"[FAIL] {src}\n{info}\n")
                elif status == "skip":
                    skipped += 1

                if finished % 500 == 0:
                    print(f"[PROGRESS] done={finished}/{total} ok_or_skip={finished-bad} skip={skipped} bad={bad}")

                # 清理时间戳字典，省内存
                fut_start.pop(fut, None)
                fut2src.pop(fut, None)

            # 检查超时：把超时的 future 从 pending 里移除（不再等待）
            now = time.time()
            if pending:
                to_drop = []
                for fut in pending:
                    start_t = fut_start.get(fut, now)
                    if now - start_t > per_task_timeout_sec:
                        to_drop.append(fut)

                if to_drop:
                    for fut in to_drop:
                        src = fut2src.get(fut, "<unknown>")
                        timeouts.append(src)
                        bad += 1
                        finished += 1  # 视为已经处理完（超时失败）
                        pending.remove(fut)

                        # 试图 cancel（如果已在跑通常 cancel 不掉，但没关系）
                        try:
                            fut.cancel()
                        except Exception:
                            pass

                        print(f"[TIMEOUT] {src}  (>{per_task_timeout_sec}s)")

                    print(f"[PROGRESS] done={finished}/{total} ok_or_skip={finished-bad} skip={skipped} bad={bad}")

            # 心跳：长期 pending 时打印几个例子
            if pending and (now - last_stall_report) >= stall_report_sec:
                sample = list(pending)[:5]
                print(f"[STALL?] pending={len(pending)} examples:")
                for f in sample:
                    print("   ", fut2src.get(f, "<unknown>"))
                last_stall_report = now

    print(f"[DONE] total={total} skip={skipped} bad={bad} timeouts={len(timeouts)}")

    # 输出 timeout 清单
    if timeout_list_path is not None and timeouts:
        p = Path(timeout_list_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(timeouts) + "\n", encoding="utf-8")
        print(f"[TIMEOUT_LIST] wrote -> {timeout_list_path}")

def process_csv(csv_path: str, out_csv_path: str, workers=4, skip_if_done=True):
    df = pd.read_csv(csv_path)

    for col in IMG_COLS:
        paths = df[col].astype(str).tolist()
        new_paths = [make_new_path(p) for p in paths]

        run_parallel(paths, new_paths, workers=workers, skip_if_done=skip_if_done, per_task_timeout_sec=120,
    timeout_list_path=str(Path(NEW_ROOT) / f"timeouts_{Path(csv_path).stem}_{col}.txt"))

        # ✅ 关键：无论是否 skip，CSV 都写 NEW_ROOT 的新路径（覆盖整列）
        df[col] = new_paths

    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

def main():
    out_train = str(Path(NEW_ROOT) / "train.csv")
    out_test  = str(Path(NEW_ROOT) / "test.csv")

    process_csv(TRAIN_CSV, out_train, workers=4, skip_if_done=True)
    process_csv(TEST_CSV,  out_test,  workers=4, skip_if_done=True)

if __name__ == "__main__":
    main()
