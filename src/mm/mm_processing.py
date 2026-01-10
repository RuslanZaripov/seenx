import os
import cv2
import torch
import imageio
import numpy as np
import torchaudio.compliance.kaldi as ta_kaldi
from subprocess import CalledProcessError, run
from PIL import Image
from decord import VideoReader, cpu
from mm.mm_constants import NUM_FRAMES, MAX_FRAMES


def expand2square(pil_img, background_color):
    """Expand PIL image to a square image by padding the shorter side."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def frame_sample(duration, mode="uniform", num_frames=None, fps=None):
    if mode == "uniform":
        assert (
            num_frames is not None
        ), "Number of frames must be provided for uniform sampling."
        # NOTE: v1 version
        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(duration - 1) / num_frames  # seg_size in frames is float

        frame_ids = []
        for i in range(num_frames):
            # Calculate the start and end indices of each segment
            start = seg_size * i
            end = seg_size * (i + 1)
            # Append the middle index of the segment to the list
            frame_ids.append((start + end) / 2)

        return np.round(np.array(frame_ids) + 1e-6).astype(int)
        # NOTE: v0 version
        # return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == "fps":
        assert fps is not None, "FPS must be provided for FPS sampling."
        segment_len = min(fps // NUM_FRAMES_PER_SECOND, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f"Unsupported frame sampling mode: {mode}")


def process_video(
    video_path,
    processor,
    s=None,
    e=None,
    aspect_ratio="pad",
    num_frames=NUM_FRAMES,
    va=False,
):
    # s and e are in seconds
    if isinstance(video_path, str):
        if s is not None and e is not None:
            s = s if s >= 0.0 else 0.0
            e = e if e >= 0.0 else 0.0
            if s > e:
                s, e = e, s
            elif s == e:
                e = s + 1

        # 1. Loading Video
        if os.path.isdir(video_path):  # Folder of frames
            frame_files = sorted(os.listdir(video_path))

            fps = 3
            num_frames_of_video = len(frame_files)
        elif video_path.endswith(".gif"):
            gif_reader = imageio.get_reader(video_path)

            fps = 25
            num_frames_of_video = len(gif_reader)
        else:
            vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)

            fps = vreader.get_avg_fps()
            num_frames_of_video = len(vreader)

        # 2. Determine frame range & Calculate frame indices
        max_frame_idx = num_frames_of_video - 1
        f_start = 0 if s is None else max(int(s * fps) - 1, 0)
        f_end = max_frame_idx if e is None else min(int(e * fps) - 1, max_frame_idx)
        frame_indices = list(range(f_start, f_end + 1))

        duration = len(frame_indices)  # duration in frames
        # 3. Sampling frame indices
        if num_frames is None:
            sampled_frame_indices = [
                frame_indices[i] for i in frame_sample(duration, mode="fps", fps=fps)
            ]
        else:
            sampled_frame_indices = [
                frame_indices[i]
                for i in frame_sample(duration, mode="uniform", num_frames=num_frames)
            ]

        # 4. Acquire frame data
        if os.path.isdir(video_path):
            video_data = [
                Image.open(os.path.join(video_path, frame_files[f_idx]))
                for f_idx in sampled_frame_indices
            ]
        elif video_path.endswith(".gif"):
            video_data = [
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB))
                for idx, frame in enumerate(gif_reader)
                if idx in sampled_frame_indices
            ]
        else:
            video_data = [
                Image.fromarray(frame)
                for frame in vreader.get_batch(sampled_frame_indices).asnumpy()
            ]

    elif isinstance(video_path, np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]

    elif isinstance(video_path, list) and isinstance(video_path[0], np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]

    elif isinstance(video_path, list) and isinstance(video_path[0], str):
        video_data = [Image.open(f) for f in video_path]

    elif isinstance(video_path, list) and isinstance(video_path[0], Image.Image):
        video_data = video_path

    else:
        raise ValueError(f"Unsupported video path type: {type(video_path)}")

    # video_data: List[Image.Image] of length num_frames from s to e seconds

    # Padding if num_frames is specified
    while num_frames is not None and len(video_data) < num_frames:
        video_data.append(
            Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8))
        )

    # MAX_FRAMES filter
    video_data = video_data[:MAX_FRAMES]

    if aspect_ratio == "pad":
        background_color = tuple(int(x * 255) for x in processor.image_mean)
        images = [expand2square(f, background_color) for f in video_data]
        video = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    else:
        images = [f for f in video_data]
        video = processor.preprocess(images, return_tensors="pt")["pixel_values"]

    if va:
        # Calculate the duration of the video in seconds
        video_duration_seconds = num_frames_of_video / fps
        # audio = process_audio_from_video(video_path, video_duration_seconds)
        audio = process_audio_from_video_range(
            video_path,
            s if s is not None else 0.0,
            e if e is not None else video_duration_seconds,
        )
        video = {"video": video, "audio": audio}

    return video


def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


def load_audio_from_video(file: str, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-i",
        file,
        "-vn",  # no video
        "-acodec",
        "pcm_s16le",  # output audio codec (pcm_s16le for .wav)
        "-ac",
        "1",  # audio channels (1 for mono)
        "-ar",
        str(sr),  # audio sample rate
        "-f",
        "s16le",  # output format (s16le for 16-bit PCM)
        "-",  # output to stdout
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0, sr


def process_audio_from_video_range(
    audio_path,
    t_start,
    t_end,
    sample_rate=16000,
    num_mel_bins=128,
):
    try:
        waveform, sr = load_audio_from_video(audio_path)
    except Exception as audio_error:
        print(f"Failed to process audio from video due to error: {audio_error}")
        waveform = torch.zeros(480000)
        waveform = waveform.numpy()
        sr = 16000

    assert 0 <= t_start < t_end, "Invalid time range"

    start = int(t_start * sr)
    end = int(t_end * sr)

    waveform = waveform[start:end]

    # convert to torch + Kaldi scaling
    waveform = torch.from_numpy(waveform).unsqueeze(0) * 2**15

    fbank = ta_kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        sample_frequency=sr,
        frame_length=25,
        frame_shift=10,
    )

    return fbank.unsqueeze(0)  # (1, T, mel)


def process_audio_from_video(
    audio_path,
    clip_duration,
    device="cpu",
    num_mel_bins=128,
    sample_rate=16000,
    clips_per_video=8,
    mean=-4.268,
    std=9.138,
):
    try:
        waveform, sr = load_audio_from_video(audio_path)
    except Exception as audio_error:
        print(f"Failed to process audio from video due to error: {audio_error}")
        waveform = torch.zeros(480000)
        waveform = waveform.numpy()
        sr = 16000

    from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=2, clips_per_video=clips_per_video
    )

    all_clips_timepoints = get_clip_timepoints(
        clip_sampler, waveform.shape[0] / sample_rate
    )

    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        s = int(clip_timepoints[0] * sample_rate)
        e = int(clip_timepoints[1] * sample_rate)
        waveform_clip = waveform[s:e]
        all_clips.append(waveform_clip)

    all_clips_tensors = [torch.from_numpy(clip) for clip in all_clips]

    wav = torch.cat(all_clips_tensors, dim=0)

    # sr - samples per second
    tgt_samples = 30 * sr  # 30 seconds
    if len(wav) > tgt_samples:
        max_start = len(wav) - tgt_samples
        start = torch.randint(0, max_start, size=(1,)).item()
        wav = wav[start : start + tgt_samples]
    if len(wav) < tgt_samples:
        pad_length = tgt_samples - len(wav)
        wav = torch.nn.functional.pad(wav, (0, pad_length), mode="constant", value=0.0)

    waveform = wav.unsqueeze(0) * 2**15

    fbank = ta_kaldi.fbank(
        waveform,
        num_mel_bins=128,
        sample_frequency=16000,
        frame_length=25,
        frame_shift=10,
    ).to(torch.bfloat16)

    return fbank.unsqueeze(0)
