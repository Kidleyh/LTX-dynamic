"""
生成 TEST_SAMPLES 的脚本

用法：
  - 把你的输入列表赋值给 `items`，每个元素格式为：
      [prompt, cond_image_path, cond_audio_path]
  - 调用 `build_test_samples(items, n_ranks=8)` 得到一个包含 8 个 rank 的字典
  - 可选：使用 `save_test_samples_py(TEST_SAMPLES, out_path)` 将结果保存为一个 python 文件（变量名为 TEST_SAMPLES）

实现细节：
  - 会把 items 保持原始顺序，平均分到 n_ranks 个子列表（前面若有余项会先分配给前几个 rank）
  - prompt_index 为全局递增（从 0 开始），确保每个样本有唯一的索引
  - cond_audio 被包装为 {"person1": audio_path}，以匹配你给出的示例格式

示例：
    items = [
        ["prompt A", "/path/to/imgA.png", "/path/to/audioA.wav"],
        ["prompt B", "/path/to/imgB.png", "/path/to/audioB.wav"],
        ...
    ]
    TS = build_test_samples(items, n_ranks=8)
    save_test_samples_py(TS, "./TEST_SAMPLES_generated.py")

"""

from typing import List, Tuple, Dict, Any
import pprint
import math
from tqdm import tqdm


def split_evenly_preserve_order(items: List[Any], n_parts: int) -> List[List[Any]]:
    """把 items 保持顺序平均分成 n_parts 份，前面的部分如果有多余项会优先得到一个额外元素。"""
    L = len(items)
    base = L // n_parts
    extras = L % n_parts
    parts = []
    idx = 0
    for i in range(n_parts):
        take = base + (1 if i < extras else 0)
        parts.append(items[idx: idx + take])
        idx += take
    return parts


def build_test_samples(items: List[Tuple[str, str, str]], n_ranks: int = 8, rank_prefix: str = "rank") -> Dict[str, List[Dict[str, Any]]]:
    """
    items: list of [prompt, cond_image_path, cond_audio_path]
    返回结构与示例一致：
      {
        "rank0": [ {"prompt":..., "prompt_index":..., "cond_image":..., "cond_audio": {"person1": ...} }, ... ],
        "rank1": [...],
        ...
      }

    prompt_index 是全局按原始 items 顺序递增的（从 0 开始）。
    """
    if n_ranks <= 0:
        raise ValueError("n_ranks must be >= 1")

    parts = split_evenly_preserve_order(items, n_ranks)

    test_samples: Dict[str, List[Dict[str, Any]]] = {}
    global_idx = 0
    for rank_i, part in enumerate(parts):
        rank_name = f"{rank_prefix}{rank_i}"
        rank_list: List[Dict[str, Any]] = []
        for item in part:
            if not (isinstance(item, (list, tuple)) and len(item) >= 3):
                raise ValueError("每个 item 必须是 [prompt, cond_image_path, cond_audio_path] 的形式")
            prompt, cond_image, cond_audio = item[0], item[1], item[2]
            d = {
                "prompt": prompt,
                "prompt_index": global_idx,
                "cond_image": cond_image,
                "cond_audio": {
                    "person1": cond_audio
                }
            }
            rank_list.append(d)
            global_idx += 1
        test_samples[rank_name] = rank_list

    return test_samples


def save_test_samples_py(test_samples: Dict[str, Any], out_path: str):
    """把 test_samples 保存为一个 python 文件，文件中会直接包含 TEST_SAMPLES = <literal>。"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated TEST_SAMPLES\n")
        f.write("# encoding: utf-8\n\n")
        f.write("TEST_SAMPLES = ")
        # 使用 pprint.pformat 生成漂亮的 Python 字面量形式
        f.write(pprint.pformat(test_samples, width=120))
        f.write("\n")


# ---- 示例运行：从文件夹自动构建 items 并生成 TEST_SAMPLES ----
# 新增函数：build_items_from_folders

# def build_items_from_folders(folder_paths,
#                              audio_sub='audios',
#                              img_sub='imgs',
#                              prompt_sub='prompts',
#                              audio_exts=('.wav',),
#                              img_exts=('.jpg', '.jpeg', '.png')):
#     """
#     从若干个父文件夹构建 items 列表。
#     每个父文件夹下应包含三子文件夹：audio_sub, img_sub, prompt_sub。
#     子文件夹内的文件通过文件名前缀（stem）一一对应，例如：
#         audios/0001.wav  imgs/0001.jpg  prompts/0001.txt
#     prompts 的 txt 文件会读取其**第一行**作为 prompt 文本。

#     返回值：items 列表，元素为 [prompt_text, img_path, audio_path]
#     """
#     from pathlib import Path
#     import os

#     items = []
#     for folder in folder_paths:
#         audio_dir = os.path.join(folder, audio_sub)
#         img_dir = os.path.join(folder, img_sub)
#         prompt_dir = os.path.join(folder, prompt_sub)

#         if not (os.path.isdir(audio_dir) and os.path.isdir(img_dir) and os.path.isdir(prompt_dir)):
#             print(f"[WARN] skip '{folder}': required subdirs not found -> {audio_dir}, {img_dir}, {prompt_dir}")
#             continue

#         # 构建 stem->path 映射
#         aud_map = {}
#         for fn in os.listdir(audio_dir):
#             p = Path(fn)
#             if p.suffix.lower() in audio_exts:
#                 aud_map[p.stem] = os.path.join(audio_dir, fn)

#         img_map = {}
#         for fn in os.listdir(img_dir):
#             p = Path(fn)
#             if p.suffix.lower() in img_exts:
#                 img_map[p.stem] = os.path.join(img_dir, fn)

#         prompt_map = {}
#         for fn in os.listdir(prompt_dir):
#             p = Path(fn)
#             if p.suffix.lower() == '.txt':
#                 prompt_map[p.stem] = os.path.join(prompt_dir, fn)

#         # 取交集并按字典序排序，确保顺序稳定
#         common_stems = sorted(set(aud_map.keys()) & set(img_map.keys()) & set(prompt_map.keys()))
#         if not common_stems:
#             print(f"[WARN] no matching samples found in '{folder}'")
#             continue

#         for stem in common_stems:
#             prompt_path = prompt_map[stem]
#             try:
#                 with open(prompt_path, 'r', encoding='utf-8') as f:
#                     first_line = f.readline().strip()
#             except Exception as e:
#                 print(f"[WARN] failed to read prompt file '{prompt_path}': {e}")
#                 first_line = ''

#             if not first_line:
#                 print(f"[WARN] empty prompt in '{prompt_path}', skip sample '{stem}'")
#                 continue

#             items.append([first_line, img_map[stem], aud_map[stem]])

#     return items

def build_items_from_folders(folder_paths,
                             audio_sub='audios',
                             img_sub='imgs',
                             prompt_sub='prompts',
                             audio_exts=('.wav',),
                             img_exts=('.jpg', '.jpeg', '.png'),
                             min_audio_sec=5.2):     # 新增：音频最短秒数
    """
    从若干个父文件夹构建 items 列表。
    只有音频时长 > min_audio_sec 才加入 items。
    """
    from pathlib import Path
    import os
    from pydub import AudioSegment    # 用于读音频时长

    items = []
    for folder in folder_paths:
        audio_dir = os.path.join(folder, audio_sub)
        img_dir = os.path.join(folder, img_sub)
        prompt_dir = os.path.join(folder, prompt_sub)

        if not (os.path.isdir(audio_dir) and os.path.isdir(img_dir) and os.path.isdir(prompt_dir)):
            print(f"[WARN] skip '{folder}': required subdirs not found -> {audio_dir}, {img_dir}, {prompt_dir}")
            continue

        # ---- 构建 stem->path 映射 ----
        aud_map = {}
        for fn in os.listdir(audio_dir):
            p = Path(fn)
            if p.suffix.lower() in audio_exts:
                aud_map[p.stem] = os.path.join(audio_dir, fn)

        img_map = {}
        for fn in os.listdir(img_dir):
            p = Path(fn)
            if p.suffix.lower() in img_exts:
                img_map[p.stem] = os.path.join(img_dir, fn)

        prompt_map = {}
        for fn in os.listdir(prompt_dir):
            p = Path(fn)
            if p.suffix.lower() == '.txt':
                prompt_map[p.stem] = os.path.join(prompt_dir, fn)

        # ---- 匹配三者交集 ----
        common_stems = sorted(set(aud_map.keys()) & set(img_map.keys()) & set(prompt_map.keys()))
        if not common_stems:
            print(f"[WARN] no matching samples found in '{folder}'")
            continue

        # ---- 逐条构建 item ----
        for stem in tqdm(common_stems, desc="building items: "):
            prompt_path = prompt_map[stem]
            try:
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
            except Exception as e:
                print(f"[WARN] failed to read prompt file '{prompt_path}': {e}")
                first_line = ''

            if not first_line:
                print(f"[WARN] empty prompt in '{prompt_path}', skip sample '{stem}'")
                continue

            audio_path = aud_map[stem]

            # ---- 读取音频长度，过滤短音频 ----
            try:
                audio = AudioSegment.from_file(audio_path)
                duration_sec = len(audio) / 1000.0
            except Exception as e:
                print(f"[WARN] cannot read audio '{audio_path}': {e}")
                continue

            if duration_sec <= min_audio_sec:
                print(f"[INFO] skip '{stem}': audio {duration_sec:.2f}s <= {min_audio_sec}s")
                continue

            # ---- 通过所有检查才加入 ----
            items.append([first_line, img_map[stem], audio_path])

    return items

if __name__ == "__main__":
    # 示例：把要扫描的 K 个父文件夹放到这里
    # 每个父文件夹应包含三个子文件夹：audios, imgs, prompts
    dataset_name = "gd_shuping_1_1"
    sample_folders = [
        # 替换为你的实际路径，例如：
        f"/gemini/platform/public/aigc/human_guozz2/code/hys/data/ode_input_data_16fps_shuping1_1209/{dataset_name}",
    ]
    min_audio_sec = 5.2

    # 从文件夹构建 items_example
    items_example = build_items_from_folders(sample_folders,
                                             audio_sub='audios',
                                             img_sub='imgs',
                                             prompt_sub='prompts',
                                             audio_exts=('.wav', '.mp3'),
                                             img_exts=('.jpg', '.jpeg', '.png'),
                                             min_audio_sec=min_audio_sec)

    print(f"Found {len(items_example)} samples from {len(sample_folders)} folders")

    # 把这些 samples 填入 TEST_SAMPLES（平均分成 8 个 rank）并保存到文件
    TS = build_test_samples(items_example, n_ranks=8)
    save_test_samples_py(TS, f"ode_samples_16fps_{dataset_name}.py")
    print("Saved TEST_SAMPLES_generated.py with", sum(len(v) for v in TS.values()), "items")
