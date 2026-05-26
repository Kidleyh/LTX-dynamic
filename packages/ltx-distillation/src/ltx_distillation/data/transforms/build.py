from ..registry import Registry, build_module
from .video_transform import VideoTransform
from .prompt_transform import PromptTransform
from .prompt_transformv2 import (
    PromptToClipEmbedding,
    PromptToTransformerEmbedding,
    PromptGenerator,
)
from .video_transformv2 import (
    SampleImages, 
    GenerateRefImages, 
    GenerateFirstRefImage, 
    GenerateAudioFeatures, 
    SampleFaceBoundingBoxes, 
    SampleLipMasks, 
    GenerateAudioFeaturesSonic,
    GenerateWav2vec2FeatureOnline,
    GenerateWhisperFeature,
    GenerateWav2vecFeature,
    GenerateChineseWav2vec2FeatureOnline,
    GenerateChineseWav2vecFeature,
    GuidanceDropSelector,
    GenerateMotionIndexes,
    SampleFaceBoundingBoxesDummy,
)
from .formatting import PackInputs

# datasets & transforms need to register
TRANSFORMS = Registry()
TRANSFORMS.register_module(VideoTransform)
TRANSFORMS.register_module(PromptTransform)
TRANSFORMS.register_module(PromptToClipEmbedding)
TRANSFORMS.register_module(PromptToTransformerEmbedding)
TRANSFORMS.register_module(PromptGenerator)
TRANSFORMS.register_module(SampleImages)
TRANSFORMS.register_module(PackInputs)
TRANSFORMS.register_module(GenerateRefImages)
TRANSFORMS.register_module(GenerateFirstRefImage)
TRANSFORMS.register_module(GenerateAudioFeatures)
TRANSFORMS.register_module(SampleFaceBoundingBoxes)
TRANSFORMS.register_module(SampleLipMasks)
TRANSFORMS.register_module(GenerateAudioFeaturesSonic)
TRANSFORMS.register_module(GenerateWav2vec2FeatureOnline)
TRANSFORMS.register_module(GenerateWhisperFeature)
TRANSFORMS.register_module(GenerateWav2vecFeature)
TRANSFORMS.register_module(GenerateChineseWav2vec2FeatureOnline)
TRANSFORMS.register_module(GenerateChineseWav2vecFeature)
TRANSFORMS.register_module(GuidanceDropSelector)
TRANSFORMS.register_module(GenerateMotionIndexes)
TRANSFORMS.register_module(SampleFaceBoundingBoxesDummy)

def build_transform(params_or_type, *args, **kwargs):
    return build_module(TRANSFORMS, params_or_type, *args, **kwargs)
