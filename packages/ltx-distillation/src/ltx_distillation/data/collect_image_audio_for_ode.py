#!/usr/bin/env python3
import os
import random
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm",
    ".mpg", ".mpeg", ".ts", ".wmv", ".m4v"
}

def _sanitize_label(s: str) -> str:
    s = s.strip().replace("@", "_").replace(" ", "_")
    s = s.replace("/", "_").replace("\\", "_")
    return s or "root"

def _find_videos(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS])

def _ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg 未找到，请先安装")

def get_video_duration(video_path: Path) -> float:
    """获取视频时长（秒）"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return float(out.decode().strip())
    except:
        return 0.0

def process_single_video(
    vid: Path,
    folder_label: str,
    out_img_dir: Path,
    out_aud_dir: Path,
    overwrite: bool,
    random_frame: bool,
    seed: int | None,
):
    """处理单个视频：随机帧 + 音频提取"""

    if seed is not None:
        random.seed(seed + hash(str(vid)))

    base_name = vid.stem.replace("@", "_").replace(" ", "_")
    img_path = out_img_dir / f"{folder_label}@@{base_name}.png"
    aud_path = out_aud_dir / f"{folder_label}@@{base_name}.wav"

    results = {
        "img": None, "aud": None, "error": None, "skip_audio": None
    }

    # ============ 🎞 抽随机帧 ============
    if img_path.exists() and not overwrite:
        results["img"] = "skip_exist"
    else:
        timestamp = 0
        if random_frame:
            dur = get_video_duration(vid)
            if dur > 0:
                timestamp = random.uniform(0, max(0, dur - 0.05))  # 避免超过末尾

        cmd_img = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-ss", f"{timestamp:.3f}",  # 随机时间点
            "-i", str(vid),
            "-frames:v", "1",
            "-q:v", "2",
            str(img_path)
        ]
        try:
            res = subprocess.run(cmd_img, check=False)
            if res.returncode == 0 and img_path.exists():
                results["img"] = str(img_path)
            else:
                results["error"] = f"extract frame failed: {vid}"
                if img_path.exists():
                    img_path.unlink()
        except Exception as e:
            results["error"] = f"frame exception: {vid} -> {e}"

    # ============ 🔊 提取音频 ============
    if aud_path.exists() and not overwrite:
        results["aud"] = "skip_exist"
    else:
        cmd_aud = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-i", str(vid),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(aud_path)
        ]
        try:
            res = subprocess.run(cmd_aud, check=False)
            if res.returncode == 0 and aud_path.exists():
                results["aud"] = str(aud_path)
            else:
                results["skip_audio"] = str(vid)
                if aud_path.exists():
                    aud_path.unlink()
        except Exception as e:
            results["error"] = f"audio exception: {vid} -> {e}"
            if aud_path.exists():
                aud_path.unlink()

    return results


def extract_random_frames_multithread(
    selections: List[Tuple[str, int]],
    output_images_folder: str,
    output_audio_folder: str,
    threads: int = 8,
    seed: int | None = None,
    overwrite: bool = False,
    random_frame: bool = True,
):
    """主函数：多线程执行所有视频的抽帧+抽音频"""

    _ensure_ffmpeg()

    out_img_dir = Path(output_images_folder)
    out_aud_dir = Path(output_audio_folder)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_aud_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "images": [],
        "audios": [],
        "skip_audio": [],
        "errors": [],
    }

    tasks = []
    executor = ThreadPoolExecutor(max_workers=threads)

    # print(selections)
    folder_path, n, prompt_folder_path = selections
    # for folder_path, n, prompt_folder_path in selections:
    folder = Path(folder_path)
    folder_label = _sanitize_label(folder.name)
    vids = _find_videos(folder)

    if not vids:
        summary["errors"].append(f"no videos in {folder}")
        # continue
        return summary

    # 随机选 n 个
    selected = vids if n >= len(vids) else random.sample(vids, n)

    for v in selected:
        tasks.append(
            executor.submit(
                process_single_video,
                v, folder_label,
                out_img_dir, out_aud_dir,
                overwrite, random_frame, seed
            )
        )

    # 等待全部线程结束
    for future in as_completed(tasks):
        res = future.result()
        if res["img"] and res["img"] != "skip_exist":
            summary["images"].append(res["img"])
        if res["aud"] and res["aud"] != "skip_exist":
            summary["audios"].append(res["aud"])
        if res["skip_audio"]:
            summary["skip_audio"].append(res["skip_audio"])
        if res["error"]:
            summary["errors"].append(res["error"])

    executor.shutdown(wait=True)
    return summary

import os
import pickle
import json

def load_data_file(pkl_or_json_path):
    """
    尝试读取 .pkl 或 .json 文件：
      - 如果是 pickle（二进制 pickle），用 pickle.load；
      - 否则尝试作为 json，用 json.load。
    返回解析后的数据对象（通常 dict），出错返回 None。
    """
    # 先尝试以二进制 pickle 读取
    try:
        with open(pkl_or_json_path, 'rb') as f:
            # 检查是否是 pickle 文件：pickle binary 协议通常以 0x80 开头。:contentReference[oaicite:0]{index=0}
            header = f.read(2)
            f.seek(0)
            if len(header) >= 2 and header[0] == 0x80:
                return pickle.load(f)
    except Exception:
        pass

    # 如果不是 pickle，尝试 json
    try:
        with open(pkl_or_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def process(src_dir, meta_dir, dst_txt_dir, valid_exts=None):
    os.makedirs(dst_txt_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        full = os.path.join(src_dir, fname)
        if not os.path.isfile(full):
            continue

        name, ext = os.path.splitext(fname)
        if '@@' not in name:
            print(f"Skipping {fname}, no '@@'")
            continue
        prefix, key = name.split('@@', 1)

        # 构造对应 meta 文件路径（试 .pkl 和 .json）
        pkl_path = os.path.join(meta_dir, key + '.pkl')
        json_path = os.path.join(meta_dir, key + '.json')
        data = None

        if os.path.isfile(pkl_path):
            data = load_data_file(pkl_path)
        elif os.path.isfile(json_path):
            data = load_data_file(json_path)
        else:
            print(f"No meta file for key {key}, skip")
            continue

        if not isinstance(data, dict):
            print(f"Meta file {key} loaded but not dict, skip")
            continue

        # 尝试两种情况
        # 如果有 prompt_panda & prompt_cogvlm2，就按之前逻辑
        if 'prompt_panda' in data and 'prompt_cogvlm2' in data:
            try:
                p1 = data['prompt_panda']['prompt'][0]
                p2 = data['prompt_cogvlm2']['prompt'][0]
            except Exception as e:
                print(f"Meta {key}: missing subkeys {e}, skip")
                continue
        # 否则尝试 short_caption / video_caption
        elif 'short_caption' in data and 'video_caption' in data:
            p1 = data['short_caption'][0]
            p2 = data['video_caption'][0]
        else:
            print(f"Meta {key} has neither prompt_panda/cogvlm2 nor short_caption/video_caption, skip")
            continue

        # 写 txt
        txt_name = f"{prefix}@@{key}.txt"
        txt_path = os.path.join(dst_txt_dir, txt_name)
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(str(p1) + "\n")
                f.write(str(p2) + "\n")
            print(f"Wrote {txt_path}")
        except Exception as e:
            print(f"Error writing {txt_path}: {e}")


if __name__ == "__main__":

    # ！！！！！！[NOTE] ！！！！！！！！！！暂时没实现读取prompt并保存，下次使用时实现一下

    data_selections = {
        "sora2pro":
        [
            [
                "/gemini/platform/public/aigc/human_guozz2/data/ai2v/zhihaowang/synthetic_processed/synthetic_sora2pro_1114/v0.0.1/video_25fps/data", 
                18000,
                "/gemini/platform/public/aigc/human_guozz2/data/ai2v/zhihaowang/synthetic_processed/synthetic_sora2pro_1114/v0.0.1//gemini_caption_ai2v/data"
            ],
            [
                "/gemini/platform/public/aigc/human_guozz2/code/hys/data/ode_input_data_16fps_shuping1_1209/sora2pro1_3/imgs",
                "/gemini/platform/public/aigc/human_guozz2/code/hys/data/ode_input_data_16fps_shuping1_1209/sora2pro1_3/audios",
                "/gemini/platform/public/aigc/human_guozz2/code/hys/data/ode_input_data_16fps_shuping1_1209/sora2pro1_3/prompts",
            ]
        ],
    }

    # selections = [
    #     # ["/gemini/platform/public/aigc/human_guozz2/data/ai2v/zhihaowang/synthetic_processed/synthetic_sora2pro_1114/v0.0.1/video_25fps/data", 10000],
    #     # ["/gemini/platform/public/aigc/human_guozz2/data/ai2v/zhihaowang/guangdian_talkinghead/guangdian_shuping_1/000/v0.0.1/video_25fps/data", 5000],
    #     # ["/gemini/platform/public/aigc/human_guozz2/data/ai2v/zhihaowang/guangdian_talkinghead/guangdian_hengping_1/v0.0.1/video_25fps/data", 5000],
    #     ,
    # ]
    for k, v in data_selections.items():
        summary = extract_random_frames_multithread(
            v[0],
            output_images_folder=v[1][0],
            output_audio_folder=v[1][1],
            threads=12,
            seed=123,
            overwrite=False,
            random_frame=False,
        )

        print("\n=== Summary ===")
        print("Images:", len(summary["images"]))
        print("Audios:", len(summary["audios"]))
        print("No-audio videos:", len(summary["skip_audio"]))
        print("Errors:", len(summary["errors"]))

        process(v[1][0], v[0][2], v[1][2])
