import re
import pandas as pd
from logger import Logger
from bs4 import BeautifulSoup

logger = Logger(show=True).get_logger('seenx')


def extract_chart_content(html_content):
    """
    Extract the chart content using the specific tag
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    chart_tag = soup.find('yta-line-chart-base', {
        'id': 'chart', 
        'class': 'style-scope yta-audience-retention-highlights-player-and-chart-v2'
    })
    
    if not chart_tag:
        raise ValueError("Could not find the chart tag with specified attributes")
    
    return str(chart_tag)


def parse_audience_retention_data(chart_content):
    """
    Extract audience retention data from SVG path element within the chart content
    """
    path_pattern = r'<path class="line-series[^"]*"[^>]*d="([^"]*)"'
    match = re.search(path_pattern, chart_content)
    
    if not match:
        raise ValueError("Could not find audience retention data path in chart content")
    
    path_data = match.group(1)
    
    points = re.findall(r'[ML](\d+),(\d+)', path_data)

    return [(int(x_str), int(y_str)) for x_str, y_str in points]


def parse_time_axis_labels(chart_content):
    """
    Extract time axis labels to determine video duration
    """
    time_pattern = r'<tspan[^>]*>(\d+):(\d+)</tspan>'
    time_matches = re.findall(time_pattern, chart_content)
    
    times = []
    for minutes, seconds in time_matches:
        total_seconds = int(minutes) * 60 + int(seconds)
        times.append(total_seconds)
    
    return times


def convert_to_time_percentage(coordinates, chart_content):
    """
    Convert pixel coordinates to time and percentage values
    """
    times = []
    percentages = []
    
    # Parse time axis to get actual duration
    time_labels = parse_time_axis_labels(chart_content)
    if time_labels:
        total_duration_seconds = max(time_labels)
    else:
        raise ValueError("Could not parse time labels")
    
    # Chart dimensions from the SVG
    chart_width = 604  # pixels (0 to 604)
    chart_height = 160  # pixels
    
    for x_px, y_px in coordinates:
        # Convert X (time) from pixels to seconds
        time_seconds = (x_px / chart_width) * total_duration_seconds
        
        # Convert Y (percentage) from pixels to percentage
        # 160px = 0%, 0px = 120% (from the axis labels)
        percentage = 120 - (y_px / chart_height) * 120
        
        times.append(time_seconds)
        percentages.append(percentage)
    
    return times, percentages, total_duration_seconds


def parse_retention(html_file_path):
    with open(html_file_path, 'r', encoding='latin-1') as f:
        html_content = f.read()

    chart_content = extract_chart_content(html_content)

    coordinates = parse_audience_retention_data(chart_content)

    times, percentages, total_duration = convert_to_time_percentage(coordinates, chart_content)
    
    logger.info(f"Extracted {len(coordinates)} from {html_file_path}")

    data = pd.DataFrame({"time": times, "value": percentages})

    data["time"] = pd.to_timedelta(data["time"], unit="s")
    data = data.set_index("time")

    # Resample to 1-second intervals and interpolate missing values
    data = data.resample("1s").mean().interpolate()

    total_seconds = int(total_duration)
    full_index = pd.to_timedelta(range(total_seconds), unit="s")
    data = data.reindex(full_index)
    
    # Forward fill any remaining NaN values
    data = data.ffill()
    
    return data
