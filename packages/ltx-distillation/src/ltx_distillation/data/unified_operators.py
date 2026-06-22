import math
import torch, torchvision, imageio, os
import imageio.v3 as iio
import json
import re
from PIL import Image
import torchaudio


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, *args, **kwargs):
        data = None
        first = True
        for operator in self.operators:
            if first:
                data = operator(*args, **kwargs)
                first = False
            else:
                data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class FrameSamplerByRateMixin:
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_rate=24, fix_frame_rate=False):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_rate = frame_rate
        self.fix_frame_rate = fix_frame_rate

    def get_reader(self, data: str):
        return imageio.get_reader(data)

    def get_available_num_frames(self, reader):
        if not self.fix_frame_rate:
            return reader.count_frames()
        meta_data = reader.get_meta_data()
        total_original_frames = int(reader.count_frames())
        duration = meta_data["duration"] if "duration" in meta_data else total_original_frames / meta_data['fps']
        total_available_frames = math.floor(duration * self.frame_rate)
        return int(total_available_frames)

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        total_frames = self.get_available_num_frames(reader)
        if int(total_frames) < num_frames:
            num_frames = total_frames
            num_frames = self.adjust_num_frames(num_frames)
        return num_frames

    def adjust_num_frames(self, num_frames: int) -> int:
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def map_single_frame_id(self, new_sequence_id: int, raw_frame_rate: float, total_raw_frames: int) -> int:
        if not self.fix_frame_rate:
            return new_sequence_id
        target_time_in_seconds = new_sequence_id / self.frame_rate
        raw_frame_index_float = target_time_in_seconds * raw_frame_rate
        frame_id = int(round(raw_frame_index_float))        
        frame_id = min(frame_id, total_raw_frames - 1)
        return frame_id


class LoadVideo(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, frame_rate=24, fix_frame_rate=False):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def __call__(self, data: str):
        reader = self.get_reader(data)
        raw_frame_rate = reader.get_meta_data()['fps']
        num_frames = self.get_num_frames(reader)
        total_raw_frames = reader.count_frames()
        frames = []
        for frame_id in range(num_frames):
            frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, total_raw_frames)
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class LoadVideoCut(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, frame_rate=24, fix_frame_rate=False):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        self.frame_processor = frame_processor

    def _crop_frame(self, frame: Image.Image, crop_box=None):
        if crop_box is None:
            return frame

        if not isinstance(crop_box, (list, tuple)) or len(crop_box) != 4:
            return frame

        x1, y1, x2, y2 = crop_box
        try:
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        except (TypeError, ValueError):
            return frame

        width, height = frame.size
        if width <= 0 or height <= 0:
            return frame

        # Support normalized xyxy boxes from label json, and tolerate absolute boxes.
        if 0.0 <= min(x1, y1, x2, y2) and max(x1, y1, x2, y2) <= 1.0:
            left = math.floor(x1 * width)
            top = math.floor(y1 * height)
            right = math.ceil(x2 * width)
            bottom = math.ceil(y2 * height)
        else:
            left = math.floor(x1)
            top = math.floor(y1)
            right = math.ceil(x2)
            bottom = math.ceil(y2)

        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))

        if right <= left or bottom <= top:
            return frame
        return frame.crop((left, top, right, bottom))

    def __call__(self, data: str, st_time=None, ed_time=None, offset=None, crop_box=None):
        reader = self.get_reader(data)
        raw_frame_rate = reader.get_meta_data()["fps"]
        total_raw_frames = reader.count_frames()
        frames = []

        offset = 0 if offset is None else int(round(float(offset)))
        st_frame = 0
        ed_frame = total_raw_frames
        if st_time is not None and ed_time is not None:
            st_frame = max(0, round(float(st_time) * raw_frame_rate))
            ed_frame = min(total_raw_frames, round(float(ed_time) * raw_frame_rate))

        # Negative offset means audio lags behind video, so drop video head frames.
        if offset < 0:
            st_frame = min(ed_frame, st_frame + abs(offset))

        available_raw_frames = max(0, ed_frame - st_frame)
        if available_raw_frames == 0:
            reader.close()
            return frames

        if self.fix_frame_rate:
            available_target_frames = math.floor(available_raw_frames / raw_frame_rate * self.frame_rate)
        else:
            available_target_frames = available_raw_frames
        num_frames = min(self.num_frames, int(available_target_frames))
        num_frames = self.adjust_num_frames(num_frames)

        for frame_id in range(num_frames):
            mapped_frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, available_raw_frames)
            mapped_frame_id = st_frame + mapped_frame_id
            frame = reader.get_data(mapped_frame_id)
            frame = Image.fromarray(frame)
            frame = self._crop_frame(frame, crop_box)
            frame = self.frame_processor(frame)
            frames.append(frame)

        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio

import torchaudio.functional as F
try:
    import whisper
except ImportError:
    whisper = None
class LoadAudioWithTorchaudio(DataProcessingOperator, FrameSamplerByRateMixin):

    def __init__(self, num_frames=121, time_division_factor=8, time_division_remainder=1, frame_rate=24, fix_frame_rate=True):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)

    def __call__(self, data: str):
        target_sr = 48000
        reader = self.get_reader(data)
        num_frames = self.get_num_frames(reader)
        duration = num_frames / self.frame_rate
        # waveform, sample_rate = torchaudio.load(data)
        # target_samples = int(duration * sample_rate)
        # current_samples = waveform.shape[-1]
        # if current_samples > target_samples:
        #     waveform = waveform[..., :target_samples]
        # elif current_samples < target_samples:
        #     padding = target_samples - current_samples
        #     waveform = torch.nn.functional.pad(waveform, (0, padding))

        # #  定义目标采样率
        # if sample_rate != target_sr:
        #     waveform = F.resample(waveform, sample_rate, target_sr)
        #     sample_rate = target_sr
        # if waveform.shape[0] == 1:
        #     waveform = waveform.repeat(1, 2, 1)

        if whisper is None:
            raise ImportError("LoadAudioWithTorchaudio requires the optional whisper package")
        audio_full = whisper.load_audio(data, sr=target_sr)
        audio_full = audio_full[: min(int(duration * target_sr), audio_full.shape[0])]
        audio = torch.from_numpy(audio_full)
        waveform = audio.unsqueeze(0).expand(2, -1)
        
        return waveform, target_sr


class LoadAudioCutWithTorchaudio(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(self, num_frames=121, time_division_factor=8, time_division_remainder=1, frame_rate=24, fix_frame_rate=True):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        self.target_sr = 48000 #51200

    def __call__(self, data: str, st_time=None, ed_time=None, offset=None):
        reader = self.get_reader(data)
        raw_frame_rate = reader.get_meta_data()["fps"]
        total_raw_frames = reader.count_frames()
        reader.close()

        if whisper is None:
            raise ImportError("LoadAudioCutWithTorchaudio requires the optional whisper package")
        waveform = torch.from_numpy(whisper.load_audio(data, sr=self.target_sr)).unsqueeze(0).expand(2, -1)
        current_samples = waveform.shape[-1]
        offset = 0 if offset is None else int(round(float(offset)))

        raw_start_frame = 0
        raw_end_frame = total_raw_frames
        if st_time is not None and ed_time is not None:
            raw_start_frame = max(0, round(float(st_time) * raw_frame_rate))
            raw_end_frame = min(total_raw_frames, round(float(ed_time) * raw_frame_rate))

        raw_start_frame = min(raw_start_frame, raw_end_frame)
        audio_start_frame = raw_start_frame + max(0, offset)
        available_raw_frames = max(0, raw_end_frame - raw_start_frame - max(0, offset) - max(0, -offset))
        if available_raw_frames <= 0:
            empty = waveform[..., :0]
            return empty, self.target_sr

        if self.fix_frame_rate:
            available_target_frames = math.floor(available_raw_frames / raw_frame_rate * self.frame_rate)
        else:
            available_target_frames = available_raw_frames
        num_frames = min(self.num_frames, int(available_target_frames))
        num_frames = self.adjust_num_frames(num_frames)
        target_duration = 0.0 if num_frames <= 0 else num_frames / self.frame_rate

        new_st_samples = math.ceil(audio_start_frame / raw_frame_rate * self.target_sr)
        new_ed_samples = new_st_samples + round(target_duration * self.target_sr)

        if new_st_samples >= 0 and new_ed_samples <= current_samples:
            waveform = waveform[..., new_st_samples:new_ed_samples]
        elif new_st_samples < 0 and new_ed_samples <= current_samples:
            waveform = torch.nn.functional.pad(waveform[..., :new_ed_samples], (-new_st_samples, 0))
        elif new_st_samples >= 0 and new_ed_samples > current_samples:
            waveform = torch.nn.functional.pad(waveform[..., new_st_samples:], (0, new_ed_samples - current_samples))
        else:
            waveform = torch.nn.functional.pad(waveform, (-new_st_samples, new_ed_samples - current_samples))

        return waveform, self.target_sr


class LoadMagiPromptFile(DataProcessingOperator):
    def __init__(self, default_prompt="The preson is talking."):
        self.default_prompt = default_prompt

    def __call__(self, data: str):
        with open(data, "r", encoding="utf-8") as f:
            payload = json.load(f)

        result = {}
        text_with_speech = (
            payload.get("audio_video_description")
            or payload.get("audiovisual_caption")
            or payload.get("video_caption")
            or self.default_prompt
        )

        speech_content = payload.get("speech_content")
        if isinstance(speech_content, dict):
            for placeholder, speech_info in speech_content.items():
                if isinstance(speech_info, dict):
                    speech_text = speech_info.get("content", "")
                else:
                    speech_text = str(speech_info)
                speech_text = speech_text.strip()
                if len(speech_text) == 0:
                    continue
                text_with_speech = text_with_speech.replace(f"[{placeholder}]", f"“{speech_text}”")

        audio_content = payload.get("audio_content")
        if isinstance(audio_content, dict):
            match_rule = r"\[.*?\]\[.*?\]:\s*\"?([^\"]+)\"?"
            for placeholder, raw_text in audio_content.items():
                if not isinstance(raw_text, str):
                    continue
                match = re.search(match_rule, raw_text)
                if match is None:
                    continue
                speech_text = match.group(1).strip()
                if len(speech_text) == 0:
                    continue
                text_with_speech = text_with_speech.replace(placeholder, f"“{speech_text}”")

        result["prompt"] = text_with_speech

        return result
