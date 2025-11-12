import re
import numpy as np
import pandas as pd
from logger import Logger
from bs4 import BeautifulSoup

logger = Logger(show=True).get_logger()


def extract_chart_content(html_content):
    """
    Extract the chart content using the specific tag
    """
    soup = BeautifulSoup(html_content, "html.parser")

    chart_tag = soup.find(
        "yta-line-chart-base",
        {
            "id": "chart",
            "class": "style-scope yta-audience-retention-highlights-player-and-chart-v2",
        },
    )

    if not chart_tag:
        raise ValueError("Could not find the chart tag with specified attributes")

    mouse_capture_pane = chart_tag.find(
        "rect", {"class": "mouseCapturePane style-scope yta-line-chart-base"}
    )

    if mouse_capture_pane:
        width = mouse_capture_pane.get("width")
        height = mouse_capture_pane.get("height")

    if not width or not height:
        raise ValueError("Could not find chart dimensions")

    return str(chart_tag), int(width), int(height)


def parse_audience_retention_data(chart_content):
    """
    Extract audience retention data from SVG path element within the chart content
    """
    path_pattern = r'<path class="line-series[^"]*"[^>]*d="([^"]*)"'
    match = re.search(path_pattern, chart_content)

    if not match:
        raise ValueError("Could not find audience retention data path in chart content")

    path_data = match.group(1)

    points = re.findall(r"[ML](\d+),(\d+)", path_data)

    return [(int(x_str), int(y_str)) for x_str, y_str in points]


def parse_time_axis_labels(chart_content):
    """
    Extract time axis labels to determine video duration
    """
    time_pattern = r"<tspan[^>]*>(\d+):(\d+)</tspan>"
    time_matches = re.findall(time_pattern, chart_content)

    times = []
    for minutes, seconds in time_matches:
        total_seconds = int(minutes) * 60 + int(seconds)
        times.append(total_seconds)

    if not times:
        raise ValueError("No time labels found in chart content")

    return times


def parse_percentage_axis_labels(chart_content):
    """
    Extract percentage axis labels to determine retention scale
    """
    percentage_pattern = r"<tspan[^>]*>(\d+)%</tspan>"
    percentage_matches = re.findall(percentage_pattern, chart_content)

    percentages = [int(pct) for pct in percentage_matches]

    if not percentages:
        raise ValueError("No percentage labels found in chart content")

    return percentages


def convert_to_time_percentage(coordinates, chart_content, width, height):
    """
    Convert pixel coordinates to time and percentage values
    """
    times = []
    percentages = []

    time_labels = parse_time_axis_labels(chart_content)
    total_duration_seconds = max(time_labels)

    percentage_labels = parse_percentage_axis_labels(chart_content)
    max_percentage = max(percentage_labels)

    logger.info(f"Chart width: {width}, Chart height: {height}")
    logger.info(
        f"Total duration: {total_duration_seconds} seconds, Max percentage: {max_percentage}%"
    )

    for x_px, y_px in coordinates:
        # Convert X (time) from pixels to seconds
        time_seconds = (x_px / width) * total_duration_seconds

        # Convert Y (percentage) from pixels to percentage
        # 0px = max_percentage (from the axis labels)
        percentage = max_percentage - (y_px / height) * max_percentage

        times.append(time_seconds)
        percentages.append(percentage)

    return times, percentages, total_duration_seconds


def parse_retention(html_file_path, points_per_second: float = 1.0):
    with open(html_file_path, "r", encoding="latin-1") as f:
        html_content = f.read()

    chart_content, width, height = extract_chart_content(html_content)

    coordinates = parse_audience_retention_data(chart_content)

    times, percentages, total_duration = convert_to_time_percentage(
        coordinates, chart_content, width, height
    )

    logger.info(f"Extracted {len(coordinates)} from {html_file_path}")

    data = pd.DataFrame({"time": times, "retention": percentages})

    # convert time column to TimedeltaIndex
    data["time"] = pd.to_timedelta(data["time"], unit="s")
    data = data.set_index("time")

    # First ensure we have second-level values and interpolate gaps
    data = data.resample("1s").mean().interpolate()

    total_seconds = int(total_duration)

    if points_per_second <= 0:
        raise ValueError("points_per_second must be > 0")

    n_samples = int(np.round(total_seconds * float(points_per_second))) + 1
    if n_samples <= 0:
        raise ValueError("points_per_second too small or total duration is zero")

    # Create high-resolution time index (seconds fractional)
    times_out = np.linspace(0.0, float(total_seconds), num=n_samples, endpoint=True)
    full_index = pd.to_timedelta(times_out, unit="s")

    # Reindex to the high-resolution grid and interpolate
    data = data.reindex(full_index)
    data = data.interpolate(method="time")

    # Fill any remaining NaNs at the edges
    data = data.ffill().bfill()

    logger.info(f"Resampled to {n_samples} samples ({points_per_second} pts/sec)")

    return data
