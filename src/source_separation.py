# NOTE: demucs should be installed

import io
import os
import shutil
import sys
import select
import subprocess as sp
from pathlib import Path
from pydub import AudioSegment
from typing import Dict, List, Tuple, Optional, IO
from .logger import Logger
from .config import Config

logger = Logger(show=True).get_logger()

# Customize the following options!
model = "htdemucs"
extensions = ["mp3", "wav", "ogg", "flac"]  # we will look for all those file types.
two_stems = None  # only separate one stems from the rest, for instance
# two_stems = "vocals"

# Options for the output audio.
mp3 = True
mp3_rate = 320
float32 = False  # output as float 32 wavs, unsused if 'mp3' is True.
int24 = False  # output as int24 wavs, unused if 'mp3' is True.
# You cannot set both `float32 = True` and `int24 = True` !!


def mp4_to_wav(input_path, output_path):
    command = f"ffmpeg -y -i {input_path} -ab 160k -ac 2 -ar 44100 -vn {output_path}"
    sp.call(command, shell=True)


def find_files(in_path):
    out = []
    for file in Path(in_path).iterdir():
        if file.suffix.lower().lstrip(".") in extensions:
            out.append(file)
    return out


def copy_process_streams(process: sp.Popen):
    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())

    while fds:
        # `select` syscall will wait until one of the file descriptors has content.
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2**16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()


def separate(files: List[str], outp: str):
    cmd = ["python3", "-m", "demucs.separate", "-o", str(outp), "-n", model]
    if mp3:
        cmd += ["--mp3", f"--mp3-bitrate={mp3_rate}"]
    if float32:
        cmd += ["--float32"]
    if int24:
        cmd += ["--int24"]
    if two_stems is not None:
        cmd += [f"--two-stems={two_stems}"]

    # files = [str(f) for f in find_files(inp)]
    # if not files:
    #     logger.info(f"No valid audio files in {inp}")
    #     return
    # logger.info("Going to separate the files:")
    # logger.info("\n".join(files))

    logger.info(f"With command: " + " ".join(cmd + files))
    p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()
    if p.returncode != 0:
        logger.info("Command failed, something went wrong.")


def combine(separate_folder: str) -> Tuple[str, str]:
    vocal_path = f"{separate_folder}/vocals.mp3"
    audio1 = AudioSegment.from_mp3(f"{separate_folder}/other.mp3")
    audio2 = AudioSegment.from_mp3(f"{separate_folder}/drums.mp3")
    audio3 = AudioSegment.from_mp3(f"{separate_folder}/bass.mp3")
    audio = audio1.overlay(audio2).overlay(audio3)
    music_path = f"{separate_folder}/mixed.mp3"
    audio.export(music_path, format="mp3")
    return music_path, vocal_path
