"""逐帧检测、聚合统计和浏览器兼容编码，不引入任务队列或 Tracking。"""
import math
import subprocess
from pathlib import Path
from time import perf_counter

import cv2

from src.services.errors import DetectionError


def transcode(ffmpeg, source, destination, audio_source=None, duration=None):
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(source)]
    if audio_source:
        command += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?"]
    else:
        command += ["-map", "0:v:0", "-map", "0:a:0?"]
    command += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart"]
    if duration:
        command += ["-t", str(duration)]
    command.append(str(destination))
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=600)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise DetectionError("视频转码失败，请确认 ffmpeg 支持 libx264 编码，并尝试其他视频。") from error
    path = Path(destination)
    if not path.is_file() or path.stat().st_size == 0:
        raise DetectionError("结果视频未能生成，请重新上传。")


def process_video(detector, source, destination, preview, intermediate, conf, config):
    started = perf_counter()
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not capture.isOpened():
            raise DetectionError("视频无法打开，请上传有效的 MP4、MOV、AVI 或 MKV 视频。")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise DetectionError("无法读取视频帧率或分辨率，请重新导出视频。")
        if width * height > config["MAX_IMAGE_PIXELS"]:
            raise DetectionError("视频分辨率过大，请降低分辨率后上传。")
        if reported_frames > config["MAX_VIDEO_FRAMES"] or reported_frames / fps > config["MAX_VIDEO_SECONDS"]:
            raise DetectionError("视频超过处理时长或帧数上限，请截取较短片段。")
        output_width, output_height = width + width % 2, height + height % 2
        writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))
        if not writer.isOpened():
            raise DetectionError("无法创建输出视频，请检查 OpenCV 编码支持和存储空间。")
        class_counts = {name: 0 for name in detector.names.values()}
        inference_time = 0.0
        confidence_sum = 0.0
        total = frames = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames += 1
            if frames > config["MAX_VIDEO_FRAMES"] or frames / fps > config["MAX_VIDEO_SECONDS"]:
                raise DetectionError("视频超过处理上限，请截取较短片段。")
            result = detector.detect_frame(frame, conf)
            inference_time += result["inference_time"]
            total += result["total"]
            confidence_sum += sum(item["confidence"] for item in result["detections"])
            for name, count in result["classes"].items():
                class_counts[name] = class_counts.get(name, 0) + count
            annotated = detector.annotate(frame, result["detections"])
            if output_width != width or output_height != height:
                annotated = cv2.copyMakeBorder(annotated, 0, output_height - height,
                                              0, output_width - width, cv2.BORDER_CONSTANT)
            writer.write(annotated)
        if frames == 0:
            raise DetectionError("视频没有可读取的画面，请换一个视频。")
        if reported_frames > 0 and frames < reported_frames:
            raise DetectionError("视频在读取过程中提前中断，文件可能已损坏，请重新导出。")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    try:
        transcode(config["FFMPEG_BINARY"], intermediate, destination, audio_source=source, duration=frames / fps)
        # 单独保留原视频，生成 H.264 预览，保证 AVI/MKV 等原文件也可在页面播放。
        transcode(config["FFMPEG_BINARY"], source, preview)
    finally:
        Path(intermediate).unlink(missing_ok=True)
    return {"total": total, "classes": class_counts, "detections": [],
            "inference_time": inference_time, "processing_time": perf_counter() - started,
            "mean_confidence": confidence_sum / total if total else None,
            "media_info": {"width": width, "height": height, "fps": fps, "frame_count": frames,
                           "duration": frames / fps, "count_mode": "sum_per_frame"}}
