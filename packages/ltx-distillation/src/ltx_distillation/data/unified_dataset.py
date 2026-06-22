from .unified_operators import *
import os
import torch, json, pandas


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        max_data_items=None,
        min_frames=1,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.max_data_items = max_data_items
        self.min_frames = min_frames
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)

    def _normalize_metadata_paths(self, metadata_path):
        if metadata_path is None:
            return []
        if isinstance(metadata_path, (list, tuple)):
            return [path for path in metadata_path if isinstance(path, str) and len(path.strip()) > 0]
        if isinstance(metadata_path, str):
            if "," in metadata_path:
                return [path.strip() for path in metadata_path.split(",") if len(path.strip()) > 0]
            if len(metadata_path.strip()) > 0:
                return [metadata_path.strip()]
        return [metadata_path]

    def _print_metadata_summary(self, metadata_paths):
        print(f"[DATASET] metadata_file_count={len(metadata_paths)}")
        for single_metadata_path in metadata_paths:
            print(f"[DATASET] metadata_file={single_metadata_path}")
        print(f"[DATASET] total_metadata_records={len(self.data)}")
        print(f"[DATASET] dataset_repeat={self.repeat}")
        print(f"[DATASET] effective_dataset_size={len(self.data) * self.repeat}")
        print(f"[DATASET] min_frames_filter={self.min_frames}")

    def _normalize_metadata_record(self, record):
        if not isinstance(record, dict):
            return record

        normalized = {}
        file_path = record.get("file_path")
        if file_path is not None:
            normalized["video"] = file_path
            normalized["input_audio"] = file_path

        lipsync = record.get("lipsync")
        if isinstance(lipsync, dict):
            normalized["av_offset"] = lipsync.get("audio_video_offset_25fps", 0) or 0
        else:
            normalized["av_offset"] = 0

        for key in (
            "video",
            "input_audio",
            "prompt",
            "negative_prompt",
            "video_caption_path",
            "label_path",
            "label_json",
            "video_train_time",
            "frame_rate",
            "in_context_videos",
            "input_image",
            "av_offset",
        ):
            if key in record and record[key] is not None:
                normalized[key] = record[key]

        if len(normalized) == 0:
            return record
        return normalized

    def _append_metadata_record(self, record):
        self.data.append(self._normalize_metadata_record(record))
        return self.max_data_items is not None and len(self.data) >= self.max_data_items

    def _load_json_array_stream(self, metadata_path):
        decoder = json.JSONDecoder()
        with open(metadata_path, "r", encoding="utf-8") as f:
            buffer = ""
            in_array = False
            while True:
                chunk = f.read(1024 * 1024)
                eof = len(chunk) == 0
                buffer += chunk
                cursor = 0

                while True:
                    while cursor < len(buffer) and buffer[cursor] in " \r\n\t,":
                        cursor += 1

                    if not in_array:
                        if cursor >= len(buffer):
                            break
                        if buffer[cursor] != "[":
                            raise ValueError(f"Unsupported JSON metadata format: {metadata_path}")
                        in_array = True
                        cursor += 1
                        continue

                    while cursor < len(buffer) and buffer[cursor] in " \r\n\t,":
                        cursor += 1

                    if cursor < len(buffer) and buffer[cursor] == "]":
                        return

                    if cursor >= len(buffer):
                        break

                    try:
                        record, end = decoder.raw_decode(buffer, cursor)
                    except json.JSONDecodeError:
                        break

                    if self._append_metadata_record(record):
                        return
                    cursor = end

                buffer = buffer[cursor:]
                if eof:
                    break

    def _load_json_metadata(self, metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            while True:
                ch = f.read(1)
                if ch == "":
                    return
                if ch in " \r\n\t":
                    continue
                first_non_ws = ch
                break

        if first_non_ws == "[":
            self._load_json_array_stream(metadata_path)
            return

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if isinstance(metadata, dict):
            if isinstance(metadata.get("data"), list):
                metadata = metadata["data"]
            elif isinstance(metadata.get("items"), list):
                metadata = metadata["items"]
            else:
                metadata = [metadata]

        for record in metadata:
            if self._append_metadata_record(record):
                break
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        frame_rate=24, fix_frame_rate=False,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                    frame_rate=frame_rate, fix_frame_rate=fix_frame_rate,
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            if self.max_data_items is not None:
                self.cached_data = self.cached_data[: self.max_data_items]
            print(f"{len(self.cached_data)} cached data files found.")
        else:
            metadata_paths = self._normalize_metadata_paths(metadata_path)
            for single_metadata_path in metadata_paths:
                if single_metadata_path.endswith(".json"):
                    self._load_json_metadata(single_metadata_path)
                elif single_metadata_path.endswith(".jsonl"):
                    with open(single_metadata_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if len(line) == 0:
                                continue
                            if self._append_metadata_record(json.loads(line)):
                                self._print_metadata_summary(metadata_paths)
                                return
                else:
                    metadata = pandas.read_csv(single_metadata_path)
                    remaining = None if self.max_data_items is None else self.max_data_items - len(self.data)
                    if remaining is not None and remaining <= 0:
                        self._print_metadata_summary(metadata_paths)
                        return
                    if remaining is not None:
                        metadata = metadata.iloc[: remaining]
                    for i in range(len(metadata)):
                        if self._append_metadata_record(metadata.iloc[i].to_dict()):
                            self._print_metadata_summary(metadata_paths)
                            return
            self._print_metadata_summary(metadata_paths)

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key, operator in self.special_operator_map.items():
                if key in self.data_file_keys:
                    continue
                if key not in data:
                    continue
                processed = operator(data[key])
                if isinstance(processed, dict):
                    data.update(processed)
                else:
                    data[key] = processed
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        processed = self.special_operator_map[key](data[key])
                        if isinstance(processed, dict):
                            data.update(processed)
                        else:
                            data[key] = processed
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        elif self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True


class UnifiedCutDataset(UnifiedDataset):
    def __init__(self, *args, enable_label_bbox_crop=False, **kwargs):
        self.enable_label_bbox_crop = enable_label_bbox_crop
        self._label_crop_cache = {}
        self._label_crop_debug_counter = 0
        super().__init__(*args, **kwargs)

    def _resolve_path(self, path):
        if path is None or not isinstance(path, str):
            return path
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_path, path)

    def _load_label_payload(self, data):
        label_json = data.get("label_json")
        if isinstance(label_json, dict):
            return label_json
        if isinstance(label_json, str):
            label_json = label_json.strip()
            if len(label_json) == 0:
                return None
            if label_json.startswith("{"):
                try:
                    return json.loads(label_json)
                except json.JSONDecodeError:
                    return None
            if os.path.exists(label_json):
                label_path = label_json
            else:
                label_path = None
        else:
            label_path = data.get("label_path")

        label_path = self._resolve_path(label_path)
        if label_path is None or not os.path.exists(label_path):
            return None
        with open(label_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_xyxy_box(self, box):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            return None
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _intersect_boxes(self, box_a, box_b):
        box_a = self._normalize_xyxy_box(box_a)
        box_b = self._normalize_xyxy_box(box_b)
        if box_a is None:
            return box_b
        if box_b is None:
            return box_a

        intersection = [
            max(box_a[0], box_b[0]),
            max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]),
            min(box_a[3], box_b[3]),
        ]
        return self._normalize_xyxy_box(intersection)

    def _get_label_intersection_crop_box(self, data):
        if not self.enable_label_bbox_crop:
            return None

        cache_key = data.get("label_path")
        if cache_key is None:
            inline_label_json = data.get("label_json")
            if isinstance(inline_label_json, str):
                cache_key = inline_label_json
            elif isinstance(inline_label_json, dict):
                cache_key = json.dumps(inline_label_json, sort_keys=True, ensure_ascii=False)
        if cache_key in self._label_crop_cache:
            return self._label_crop_cache[cache_key]

        crop_box = None
        label_payload = self._load_label_payload(data)
        if isinstance(label_payload, dict):
            black_box = label_payload.get("black")
            craft = label_payload.get("craft")
            cropped_coord = craft.get("cropped_coord") if isinstance(craft, dict) else None
            crop_box = self._intersect_boxes(black_box, cropped_coord)

        self._label_crop_cache[cache_key] = crop_box
        if crop_box is not None and self._label_crop_debug_counter < 5:
            print(
                f"[DATASET][label_crop] label={data.get('label_path', '<inline_label_json>')} "
                f"crop_box={crop_box}"
            )
            self._label_crop_debug_counter += 1
        return crop_box

    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        frame_rate=24, fix_frame_rate=False,
    ):
        return LoadVideoCut(
            num_frames,
            time_division_factor,
            time_division_remainder,
            frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
            frame_rate=frame_rate,
            fix_frame_rate=fix_frame_rate,
        )

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            return self.cached_data_operator(data)


        try:
            data = self.data[data_id % len(self.data)].copy()

            for key, operator in self.special_operator_map.items():
                if key in self.data_file_keys:
                    continue
                if key not in data:
                    continue
                processed = operator(data[key])
                if isinstance(processed, dict):
                    data.update(processed)
                else:
                    data[key] = processed

            video_source = data.get("video")
            if video_source is not None:
                crop_box = self._get_label_intersection_crop_box(data)
                data["video"] = self.main_data_operator(
                    self._resolve_path(video_source),
                    data.get("st_time"),
                    data.get("ed_time"),
                    -(data.get("av_offset") or 0),
                    crop_box,
                )
                if len(data["video"]) < self.min_frames:
                    return None
                if hasattr(self.main_data_operator, "frame_rate"):
                    data["frame_rate"] = self.main_data_operator.frame_rate

            if "input_audio" in self.special_operator_map and data.get("input_audio") is not None:
                data["input_audio"] = self.special_operator_map["input_audio"](
                    self._resolve_path(data["input_audio"]),
                    data.get("st_time"),
                    data.get("ed_time"),
                    -(data.get("av_offset") or 0),
                )

            for key in self.data_file_keys:
                if key in ("video", "input_audio"):
                    continue
                if key not in data:
                    continue
                if key in self.special_operator_map:
                    processed = self.special_operator_map[key](data[key])
                    if isinstance(processed, dict):
                        data.update(processed)
                    else:
                        data[key] = processed
                else:
                    data[key] = self.main_data_operator(data[key])
            return data
        except Exception as e:
            import logging
            logging.warning(f"Error loading data_id {data_id}: {e}. Trying next index.")
            # 递归调用下一个 id，确保 DataLoader 不会中断
            return self.__getitem__(data_id + 1)
