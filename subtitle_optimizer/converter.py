"""
Subtitle format conversion utilities
Converts SRT to WebVTT format for browser-native playback
Parses SRT files into transcript segments
"""

import re


def srt_to_vtt(srt_content: str) -> str:
    """
    Convert SRT subtitle content to WebVTT format

    Args:
        srt_content: String content of an SRT file

    Returns:
        String content in WebVTT format
    """
    # Start with WebVTT header
    vtt_content = "WEBVTT\n\n"

    # SRT uses commas for milliseconds, VTT uses periods
    # SRT: 00:00:01,500 --> 00:00:04,000
    # VTT: 00:00:01.500 --> 00:00:04.000

    # Replace comma with period in timestamps
    srt_content = re.sub(r"(\d{2}:\d{2}:\d{2}),(\d{3})", r"\1.\2", srt_content)

    # Remove sequence numbers (VTT doesn't require them, though they're allowed)
    # Split by double newlines to get subtitle blocks
    blocks = re.split(r"\n\s*\n", srt_content.strip())

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split("\n")

        # First line is typically the sequence number in SRT - skip it if it's just a number
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]

        # Rest is the subtitle content
        if lines:
            vtt_content += "\n".join(lines) + "\n\n"

    return vtt_content.strip()


def srt_file_to_vtt_file(srt_filepath: str, vtt_filepath: str) -> None:
    """
    Convert an SRT file to a WebVTT file

    Args:
        srt_filepath: Path to input SRT file
        vtt_filepath: Path to output VTT file
    """
    with open(srt_filepath, "r", encoding="utf-8") as f:
        srt_content = f.read()

    vtt_content = srt_to_vtt(srt_content)

    with open(vtt_filepath, "w", encoding="utf-8") as f:
        f.write(vtt_content)


def parse_srt_to_segments(srt_content: str) -> tuple[list[dict], str]:
    """
    Parse SRT content into transcript segments format

    Args:
        srt_content: String content of an SRT file

    Returns:
        Tuple of (segments_list, full_text)
        - segments_list: List of dicts with 'start', 'end', 'text' keys
        - full_text: Combined text of all segments
    """
    segments = []
    texts = []

    # Split by double newlines to get subtitle blocks
    blocks = re.split(r"\n\s*\n", srt_content.strip())

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split("\n")

        # Skip if not enough lines
        if len(lines) < 2:
            continue

        # First line should be sequence number
        if not lines[0].strip().isdigit():
            continue

        # Second line should be timestamp
        timestamp_pattern = (
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
        )
        timestamp_match = re.match(timestamp_pattern, lines[1])

        if not timestamp_match:
            continue

        # Parse start time
        start_h, start_m, start_s, start_ms = timestamp_match.groups()[:4]
        start_time = (
            int(start_h) * 3600
            + int(start_m) * 60
            + int(start_s)
            + int(start_ms) / 1000.0
        )

        # Parse end time
        end_h, end_m, end_s, end_ms = timestamp_match.groups()[4:]
        end_time = (
            int(end_h) * 3600 + int(end_m) * 60 + int(end_s) + int(end_ms) / 1000.0
        )

        # Remaining lines are the text
        text = " ".join(lines[2:]).strip()

        if text:
            segments.append({"start": start_time, "end": end_time, "text": text})
            texts.append(text)

    full_text = " ".join(texts)
    return segments, full_text
