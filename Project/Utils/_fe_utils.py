from ._progess_bar import MgProgressbar

def isurl(probe_url):
    import requests
    try:
        response = requests.get(probe_url)
        return True, ''
    except requests.ConnectionError as conn_err:
        return False, str(conn_err)
    except requests.exceptions.MissingSchema as miss_err:
        return False, str(miss_err)
    except Exception as generic_err:
        return False, str(generic_err)


def str2sec(time_string):
    """
    Converts a time code string into seconds.
    Args:
        time_string (str): The time code to convert. Eg. '01:33:42'.
    Returns:
        float: The time code converted to seconds.
    """
    elems = [float(elem) for elem in time_string.split(':')]
    return elems[0]*3600 + elems[1]*60 + elems[2]


class FFdownladError(Exception):
    def __init__(self, message):
        self.message = message


def download_video_ffmpeg(url, target_name=None):
    """
    Downloads a video from the web using ffmpeg.
    Args:
        url (str): Url to the input video file to convert.
        target_name (str, optional): Target filename as path. Defaults to None (which assumes that the input filename should be used).
        overwrite (bool, optional): Whether to allow overwriting existing files or to automatically increment target filename to avoid overwriting. Defaults to False.
    Returns:
        str: The path to the output '.avi' file.
    """
    #ret, err = isurl(probe_url=url)
    #if ret == False:
    #    raise FFdownladError(err)

    import os
    import pipes

    ri = url.rindex('/') + 1
    filename = url[ri:]
    of, fex = os.path.splitext(filename)
    if not target_name:
        target_name = "./" + of + fex
    else:
        target_name = target_name + fex
    cmds = ["ffmpeg", "-y", "-i", pipes.quote(url), "-c:v", "libx264",
            "-preset", "ultrafast", "-crf", "19", "-an", target_name]
    ffmpeg_cmd(cmds, get_length(pipes.quote(url)), pb_prefix='Downloading file {0}:'.format(filename))
    return target_name


def convert_to_avi(filename, target_name=None):
    """
    Converts a video to one with .avi extension using ffmpeg.
    Args:
        filename (str): Path to the input video file to convert.
        target_name (str, optional): Target filename as path. Defaults to None (which assumes that the input filename should be used).
        overwrite (bool, optional): Whether to allow overwriting existing files or to automatically increment target filename to avoid overwriting. Defaults to False.
    Returns:
        str: The path to the output '.avi' file.
    """

    import os
    of, fex = os.path.splitext(filename)
    if fex == '.avi':
        print(f'{filename} is already in avi container.')
        return filename
    if not target_name:
        target_name = of + '.avi'
    cmds = ['ffmpeg', "-y", "-i", filename, "-c:v", "mjpeg",
            "-q:v", "3", "-c:a", "copy", target_name]
    ffmpeg_cmd(cmds, get_length(filename), pb_prefix='Converting to avi:')
    return target_name


def extract_subclip(filename, t1, t2, target_name=None):
    """
    Extracts a section of the video using ffmpeg.
    Args:
        filename (str): Path to the input video file.
        t1 (float): The start of the section to extract in seconds.
        t2 (float): The end of the section to extract in seconds.
        target_name (str, optional): The name for the output file. If None, the name will be \<input name\>SUB\<start time in ms\>_\<end time in ms\>.\<file extension\>. Defaults to None.
        overwrite (bool, optional): Whether to allow overwriting existing files or to automatically increment target filename to avoid overwriting. Defaults to False.
    Returns:
        str: Path to the extracted section as a video.
    """

    import os
    import numpy as np
    name, ext = os.path.splitext(filename)
    length = get_length(filename)
    start, end = np.clip(t1, 0, length), np.clip(t2, 0, length)
    if start > end:
        end = length

    if not target_name:
        T1, T2 = [int(1000*t) for t in [start, end]]
        target_name = "%sSUB%d_%d.%s" % (name, T1, T2, ext)

    # avoiding ffmpeg glitch if format is not avi:
    if os.path.splitext(filename)[1] != '.avi':
        cmd = ['ffmpeg', "-y",
               "-ss", "%0.2f" % start,
               "-i", filename,
               "-t", "%0.2f" % (end-start),
               "-map", "0", target_name]
    else:
        cmd = ['ffmpeg', "-y",
               "-ss", "%0.2f" % start,
               "-i", filename,
               "-t", "%0.2f" % (end-start),
               "-map", "0", "-codec", "copy", target_name]

    ffmpeg_cmd(cmd, length, pb_prefix='Trimming:')
    return target_name 


class FFprobeError(Exception):
    def __init__(self, message):
        self.message = message


def ffprobe(filename):
    """
    Returns info about video/audio file using FFprobe.
    Args:
        filename (str): Path to the video file to measure.
    Returns:
        str: decoded FFprobe output (stdout) as one string.
    """
    import subprocess
    command = ['ffprobe', filename]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    try:
        out, err = process.communicate(timeout=10)
    except TimeoutExpired:
        process.kill()
        out, err = process.communicate()

    if err:
        raise FFprobeError(err)
    else:
        if out.splitlines()[-1].find("No such file or directory") != -1:
            raise FileNotFoundError(out.splitlines()[-1])
        else:
            return out 


def get_length(filename):
    """
    Gets the length (in seconds) of a video using FFprobe.
    Args:
        filename (str): Path to the video file to measure.
    Returns:
        float: The length of the input video file in seconds.
    """
    out = ffprobe(filename)
    out_array = out.splitlines()
    duration = None
    at_line = -1
    while duration == None:
        duration = out_array[at_line] if out_array[at_line].find(
            "Duration:") != -1 else None
        at_line -= 1
        if at_line < -len(out_array):
            raise NoDurationError(
                "Could not get duration.")
    duration_array = duration.split(' ')
    time_string_index = duration_array.index("Duration:") + 1
    time_string = duration_array[time_string_index][:-1]
    elems = [float(elem) for elem in time_string.split(':')]
    return elems[0]*3600 + elems[1]*60 + elems[2]


def get_framecount(filename, fast=True):
    """
    Returns the number of frames in a video using FFprobe.
    Args:
        filename (str): Path to the video file to measure.
    Returns:
        int: The number of frames in the input video file.
    """
    import subprocess
    command_query_container = 'ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1'.split(
        ' ')
    command_query_container.append(filename)
    command_count = 'ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1'.split(
        ' ')
    command_count.append(filename)
    command = command_query_container if fast else command_count

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    try:
        out, err = process.communicate(timeout=10)
    except TimeoutExpired:
        process.kill()
        out, err = process.communicate()

    if err:
        raise FFprobeError(err)

    elif out:
        if out.splitlines()[-1].find("No such file or directory") != -1:
            raise FileNotFoundError(out.splitlines()[-1])
        elif out.startswith("N/A"):
            if fast:
                return get_framecount(filename, fast=False)
            else:
                raise FFprobeError(
                    "Could not count frames. (Is this a video file?)")
        else:
            return int(out)

    else:
        if fast:
            return get_framecount(filename, fast=False)
        else:
            raise FFprobeError(
                "Could not count frames. (Is this a video file?)")


def get_fps(filename):
    """
    Gets the FPS (frames per second) value of a video using FFprobe.
    Args:
        filename (str): Path to the video file to measure.
    Returns:
        float: The FPS value of the input video file.
    """
    out = ffprobe(filename)
    out_array = out.splitlines()
    video_stream = None
    at_line = -1
    while video_stream == None:
        video_stream = out_array[at_line] if out_array[at_line].find(
            "Video:") != -1 else None
        at_line -= 1
        if at_line < -len(out_array):
            raise NoStreamError(
                "No video stream found. (Is this a video file?)")
    video_stream_array = video_stream.split(',')
    fps = None
    at_chunk = -1
    while fps == None:
        fps = float(video_stream_array[at_chunk].split(
            ' ')[-2]) if video_stream_array[at_chunk].split(' ')[-1] == 'fps' else None
        at_chunk -= 1
        if at_chunk < -len(video_stream_array):
            raise FFprobeError("Could not fetch FPS.")
    return fps


class FFmpegError(Exception):
    def __init__(self, message):
        self.message = message


def ffmpeg_cmd(command, total_time, pb_prefix='Progress', print_cmd=False, stream=True):
    """
    Run an ffmpeg command in a subprocess and show progress using an MgProgressbar.
    Args:
        command (list): The ffmpeg command to execute as a list. Eg. ['ffmpeg', '-y', '-i', 'myVid.mp4', 'myVid.mov']
        total_time (float): The length of the output. Needed mainly for the progress bar.
        pb_prefix (str, optional): The prefix for the progress bar. Defaults to 'Progress'.
        print_cmd (bool, optional): Whether to print the full ffmpeg command to the console before executing it. Good for debugging. Defaults to False.
        stream (bool, optional): Whether to have a continuous output stream or just (the last) one. Defaults to True (continuous stream).
    Raises:
        KeyboardInterrupt: If the user stops the process.
        FFmpegError: If the ffmpeg process was unsuccessful.
    """
    import subprocess
    pb = MgProgressbar(total=total_time, prefix=pb_prefix)

    # hide banner
    command = ["ffmpeg", "-hide_banner"] + command[1:]

    if print_cmd:
        print()
        if type(command) == list:
            print(' '.join(command))
        else:
            print(command)
        print()

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    returncode = None
    all_out = ''

    try:
        while True:

            if stream:
                out = process.stdout.readline()
            else:
                out = process.stdout.read()
            all_out += out
            if out == '':
                process.wait()
                returncode = process.returncode
                break
            elif out.startswith('frame='):
                out_list = out.split()
                time_ind = [elem.startswith('time=')
                            for elem in out_list].index(True)
                time_str = out_list[time_ind][5:]
                time_sec = str2sec(time_str)
                pb.progress(time_sec)
            
        if returncode in [None, 0]:
            pb.progress(total_time)
        else:
            raise FFmpegError(all_out)

    except KeyboardInterrupt:
        try:
            process.terminate()
        except OSError:
            pass
        process.wait()
        raise KeyboardInterrupt