import os
import io
import pipes
import hashlib
import joblib
import requests
import librosa
import cv2
import numpy as np
import soundfile as sf

# Custom file imports
from ._progress_bar import MgProgressbar
from ._filter import filter_frame


class FFdownladError(Exception):
    def __init__(self, message):
        self.message = message


class FFprobeError(Exception):
    def __init__(self, message):
        self.message = message


class NoStreamError(FFprobeError):
    pass


class NoDurationError(FFprobeError):
    pass


class FFmpegError(Exception):
    def __init__(self, message):
        self.message = message


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


def get_id(to_hash):
    return hashlib.sha1(to_hash.encode('utf-8')).hexdigest()


def get_url_payload(url):
    ri = url.rindex('/') + 1
    filename = url[ri:]
    of, fex = os.path.splitext(filename)
    return (of, fex)


def filter_existing_urls(elems, indict):
    total_len = len(elems)
    dkeys = indict.keys()
    for idx in range(total_len):
        url = elems[idx]
        meta = get_url_payload(url)[0]
        if get_id(meta) in dkeys:
            print('Skipping element {0}'.format(meta))
            continue 
        yield url

############################################################
#                                                          #
#                                                          #
############################################################

class DatasetManager(object):

    def __init__(self, temp_dir, db_dir):
        self.tmpname = '/tempvid'
        self.datname = '/mgdb.pkl'

        self.temp_dir = temp_dir + self.tmpname
        self.db_dir = db_dir + self.datname 

        self.db = None
        self.db_len = 0
        try:
            self.db = joblib.load(self.db_dir)
            self.db_len = len(self.db.keys())
        except FileNotFoundError as e:
            self.db = {"Video": {}, "Audio": {}}
            joblib.dump(self.db, self.db_dir)

    def get_video_download_manager(self, **video_dowload_params):
        color = video_dowload_params['color']
        sizex = video_dowload_params['sizex']
        sizey = video_dowload_params['sizey']
        begt = video_dowload_params['start_t'] 
        endt = video_dowload_params['end_t']
        database = self.db["Video"]
        return DatasetVideoLoader(database, color, sizex, sizey, begt, endt, self.temp_dir)


    def get_audio_download_manager(self, **audio_dowload_params):
        samp = audio_dowload_params["samplerate"]
        begt = audio_dowload_params["start_t"]
        endt = audio_dowload_params["end_t"]
        database = self.db["Audio"]
        return DatasetAudioLoader(database, samp, begt, endt)

    def download_and_dump_audio(self, url_list, **audio_download_params):
        audio_manager = self.get_audio_download_manager(**audio_download_params)

        idx = 0
        retry = False
        try:
            for idx in audio_manager.parse_db(url_list):
                pass
        except Exception as e:
            e_str = str(e)
            if "RemoteDisconnected" in e_str:
                retry = True
                print(e_str )
            else:
                raise(e)
        
        nparsed = idx    
        if nparsed > 0:
            self.dump()

        if retry == True:
            self.download_and_dump_audio(url_list, **video_dowload_params)

    def download_and_dump_videos(self, url_list, **video_dowload_params):
        video_manager = self.get_video_download_manager(**video_dowload_params)
        dump_every_n = 10
        idx = 0
        retry = False
        try:
            for ii, idx in enumerate(video_manager.parse_db(url_list)):
                if (ii != 0) and ((ii % dump_every_n) == 0):
                    self.dump()
        except Exception as e:
            e_str = str(e)
            if "RemoteDisconnected" in e_str:
                retry = True
                print(e_str )
            else:
                raise(e)
        
        nparsed = idx    
        if nparsed > 0:
            self.dump()

        if retry == True:
            self.download_and_dump_videos(url_list, **video_dowload_params)

    def remove_entry(self, sub_db, url, dump=False):
        entry = get_id(get_url_payload(url)[0])
        if entry in self.db[sub_db]:
            print("Removing entry with id: {0}".format(entry))
            del self.db[entry]

        if dump == True:
            joblib.dump(self.db, self.db_path)

    def remove_entries(self, sub_db, url_list, dump=False):
        for url in url_list:
            self.remove_entry(sub_db, url)

        if dump == True:
            joblib.dump(self.db, self.db_path)   

    def get_db(self):
        return self.db

    def dump(self):
        print("Dumping database to {0}".format(self.db_dir))
        joblib.dump(self.db, self.db_dir, compress=3)


class DatasetAudioLoader(object):

    def __init__(self, database, sr=22050, start_t=None, end_t=None):
        self.sr = sr
        self.trim = (start_t, end_t)
        self.audio_db = database

    def parse_urls_streaming(self, url_list):
        url_iter = filter_existing_urls(url_list, self.audio_db)
        with requests.Session() as s:
            for url in url_iter:
                filp = download_audio_streaming(url, s)
                yield filp, url

    def parse_db(self, url_list):
        url_iter = self.parse_urls_streaming(url_list)
        for idx, (filp, url) in enumerate(url_iter):
            # Begin new entry
            db_entry = {}

            # Init storage for audio
            audio = None

            # Check for trimming
            if not None in self.trim:
                t1, t2 = self.trim
                offset = t1
                duration = t2 - t1
                audio, _ = librosa.load(filp, sr=self.sr, mono=True,
                    offset=offset, duration=duration)
            else:
                audio, _ = librosa.load(filp, sr=self.sr, mono=True)

            # Fetch the music id
            musicID = get_url_payload(url)[0]

            # Set entry
            db_entry["MusicID"] = musicID
            db_entry["SampleRate"] = self.sr
            db_entry["Duration"] = librosa.get_duration(y=audio, sr=self.sr)
            db_entry["RawAudio"] = audio

            # Generate a unique id and set new entry 
            id_ = get_id(musicID)
            self.audio_db[id_] = db_entry
            yield idx + 1


class DatasetVideoLoader(object):

    def __init__(self, database, color=True, sizex=640, sizey=480, start_t=None, end_t=None, temp_dir=None):
        self.color = color
        self.size = (sizex, sizey)
        self.trim = (start_t, end_t)

        self.temp_dir = temp_dir
        self.video_db = database 


    def parse_urls_streaming(self, url_list):
        url_iter = filter_existing_urls(url_list, self.video_db)
        with requests.Session() as s:
            for url in url_iter:
                target_name, original_filename =\
                    download_video_streaming(url, s, target_name=self.temp_dir)
                yield (target_name, original_filename)

    def parse_db(self, url_list):
        url_iter = self.parse_urls_streaming(url_list)
        for idx, (video_path, meta_tokens) in enumerate(url_iter):
            # Begin new entry
            db_entry = {}

            # Check for trimming
            if not None in self.trim:
                t1, t2 = self.trim
                video_path = extract_subclip(video_path, t1, t2)

            # Compute video length (s) and fps
            fps = get_fps(video_path)
            sec = get_length(video_path) 

            # Split meta tokens
            tokens = meta_tokens.strip().split("_")
            token_dict = {
                "DanceGenre": tokens[0],
                "Situation":  tokens[1],
                "CameraID":   tokens[2],
                "DancerIDs":  tokens[3:-2],
                "MusicID":    tokens[-2],
                "ChoreoID":   tokens[-1]
            }

            # Compute motiongram
            mgx, mgy = motiongram(video_path,
                size=self.size,
                color=self.color    
            )

            # Set entry
            db_entry["FrameRate"] = fps
            db_entry["Duration"] = sec
            db_entry["MetaInfo"] = token_dict
            db_entry["MotiongramX"] = mgx
            db_entry["MotiongramY"] = mgy 

            # Generate id and set new entry
            id_ = get_id(meta_tokens)
            self.video_db[id_] = db_entry
            yield idx + 1


############################################################
#                                                          #
#                                                          #
############################################################


def motiongram(filename, size, color, filtertype="regular", thresh=0.05, kernel_size=5):
    # First extract desired width and height
    width, height = size

    # Get number of frames
    length = get_framecount(filename)

    # Setup openCV video capture obj to process video-file
    # frame by frame
    vidcap = cv2.VideoCapture(filename)
    ret, frame = vidcap.read()
    prev_frame = None

    # Resize the first frame
    frame = cv2.resize(frame, (width, height))

    # Setup motiogram dimensions (containers)
    # Check for color settings before init
    gramx, gramy = None, None
    if color == True:
        gramx = np.zeros([1,  width, 3])
        gramy = np.zeros([height, 1, 3])
    else:
        gramx = np.zeros([1,  width])
        gramy = np.zeros([height, 1])

        # Convert the color of the first frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Setup another progress bar
    ii = 0
    pb = MgProgressbar(total=length, prefix="Extracting motiongram:")

    # Process frame by frame
    while vidcap.isOpened():
        #
        # POTENTIAL BLURRING HERE
        # 
        
        # Set previous frame
        prev_frame = np.copy(frame)

        # Read a new frame
        ret, frame = vidcap.read()
        if ret == False:
            pb.progress(length)
            break

        # Resize additional frames
        frame = cv2.resize(frame, (width, height))

        #
        # POTENTIAL BLURRING HERE
        #

        # Check color settings for new frame
        if color == False:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Cast "frame" into numpy explicitly and cast all
        # pixel values int 32-bit signed integers (to prevet byte crashes etc.)
        frame = np.array(frame)
        frame = frame.astype(np.int32)

        # Process frames depending on color opt
        if color == True:
            motion_frame_rgb = np.zeros([height, width, 3])

            # Process color channels independently
            # (Necessary for desired precision)
            for i in range(frame.shape[-1]):
                motion_frame = (
                    np.abs(frame[:, :, i]-prev_frame[:, :, i])).astype(np.uint8)
                motion_frame = filter_frame(
                    motion_frame, filtertype, thresh, kernel_size)
                motion_frame_rgb[:, :, i] = motion_frame

            # Compute the motiongram
            movement_y = np.mean(motion_frame_rgb, axis=1)\
                            .reshape(height, 1, 3)
            movement_x = np.mean(motion_frame_rgb, axis=0)\
                            .reshape(1, width, 3)
            
            # Append the the correct containers
            gramy = np.append(gramy, movement_y, axis=1)
            gramx = np.append(gramx, movement_x, axis=0)

        else:
            #
            # DO THE SAME AS ABOVE WITH A SINGEL CHANNEL
            #
            motion_frame = (
                np.abs(frame-prev_frame)).astype(np.uint8)
            motion_frame = filter_frame(
                motion_frame, filtertype, thresh, kernel_size)

            movement_y = np.mean(motion_frame, axis=1)\
                            .reshape(height, 1)
            movement_x = np.mean(motion_frame, axis=0)\
                            .reshape(1, width)

            gramy = np.append(gramy, movement_y, axis=1)
            gramx = np.append(gramx, movement_x, axis=0)

        # Increment progress bar and loop iterator
        pb.progress(ii)
        ii += 1

    # Post-process the computed motiongram(s)
    if color == False:
        # Normalize before converting to uint8 to keep precision
        gramx = gramx/gramx.max()*255
        gramy = gramy/gramy.max()*255
        gramx = cv2.cvtColor(gramx.astype(
            np.uint8), cv2.COLOR_GRAY2BGR)
        gramy = cv2.cvtColor(gramy.astype(
            np.uint8), cv2.COLOR_GRAY2BGR)

    gramx = (gramx-gramx.min())/(gramx.max()-gramx.min())*255.0
    gramy = (gramy-gramy.min())/(gramy.max()-gramy.min())*255.0

    # Allway equalize the motiongrams
    gramx = gramx.astype(np.uint8)
    gramx_hsv = cv2.cvtColor(gramx, cv2.COLOR_BGR2HSV)
    gramx_hsv[:, :, 2] = cv2.equalizeHist(gramx_hsv[:, :, 2])
    gramx = cv2.cvtColor(gramx_hsv, cv2.COLOR_HSV2RGB)

    gramy = gramy.astype(np.uint8)
    gramy_hsv = cv2.cvtColor(gramy, cv2.COLOR_BGR2HSV)
    gramy_hsv[:, :, 2] = cv2.equalizeHist(gramy_hsv[:, :, 2])
    gramy = cv2.cvtColor(gramy_hsv, cv2.COLOR_HSV2RGB)

    return gramx, gramy


############################################################
#                                                          #
#                                                          #
############################################################


def download_audio_streaming(url, session_obj):
    """
    Downloads an audiofile from the web using a streaming context manager.
    Args:
        url (str): Url to the input video file to convert.
        session_obj (Session): The http session to use
    Returns:
        bytesIO: a file-like pointer to bytes in memory
    """
    s = session_obj
    filp = io.BytesIO()
    of, fex = get_url_payload(url)
    with s.get(url, stream=True) as resp:
        resp.raise_for_status()
        total_bytes = int(resp.headers['content-length'])
        pb = MgProgressbar(total=total_bytes, 
            prefix='Downloading file {0}:'.format(of+fex))

        nbytes = 0
        chunk_size = 8 * 1024

        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                break
            filp.write(chunk)
            nbytes += len(chunk)
            pb.progress(nbytes)

        pb.progress(total_bytes)

    return io.BytesIO(filp.getbuffer())




def download_video_streaming(url, session_obj, target_name=None):
    """
    Downloads a video from the web using a streaming context manager.
    Args:
        url (str): Url to the input video file to convert.
        target_name (str, optional): Target filename as path. Defaults to None (which assumes that the input filename should be used).
    Returns:
        str: The path to the output file.
        str: The original filename without file-extension
    """
    of, fex = get_url_payload(url)  
    if not target_name:
        target_name = "./" + of + fex
    else:
        target_name = target_name + fex

    s = session_obj
    with s.get(url, stream=True) as resp:
        resp.raise_for_status()
        total_bytes = int(resp.headers['content-length'])
        pb = MgProgressbar(total=total_bytes, 
            prefix='Downloading file {0}:'.format(of+fex))

        nbytes = 0
        chunk_size = 16 * 1024
        with open(target_name, 'wb') as f: 
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    break
                f.write(chunk)
                nbytes += len(chunk)
                pb.progress(nbytes)

        pb.progress(total_bytes)

    target_name = convert_to_avi(target_name)
    return target_name, of


def download_video_ffmpeg(url, target_name=None):
    """
    Downloads a video from the web using ffmpeg.
    Args:
        url (str): Url to the input video file to convert.
        target_name (str, optional): Target filename as path. Defaults to None (which assumes that the input filename should be used).
        overwrite (bool, optional): Whether to allow overwriting existing files or to automatically increment target filename to avoid overwriting. Defaults to False.
    Returns:
        str: The path to the output '.avi' file.
        str: The original filename without file-extension
    """
    of, fex = get_url_payload(url)  
    if not target_name:
        target_name = "./" + of + fex
    else:
        target_name = target_name + fex
    cmds = ["ffmpeg", "-y", "-i", pipes.quote(url), "-c:v", "libx264",
            "-preset", "ultrafast", "-crf", "19", "-an", target_name]
    ffmpeg_cmd(cmds, get_length(pipes.quote(url)), pb_prefix='Downloading file {0}:'.format(of+fex))
    return target_name, of


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

    of, fex = os.path.splitext(filename)
    if fex == '.avi':
        print(f'{filename} is already in avi container.')
        return filename
    if not target_name:
        target_name = of + '.avi'
    cmds = ['ffmpeg', "-y", "-i", filename, "-c:v", "mjpeg",
            "-q:v", "3", "-c:a", "copy", "-an", target_name]
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
    except subprocess.TimeoutExpired:
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
    except subprocess.TimeoutExpired:
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