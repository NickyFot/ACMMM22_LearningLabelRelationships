import os
import sys
import subprocess
import datetime

import typing


def probe_file(filename: str) -> float:
    """Get the length of a video in seconds
    Args:
        filename (str): the path to the video file
    Returns:
        the length of the video in seconds (float)

    """
    cmnd = [
        'ffprobe',
        '-v',
        'error',
        '-show_entries',
        'format=duration',
        '-of',
        'default=noprint_wrappers=1:nokey=1',
        filename
    ]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(filename)
    out, err = p.communicate()
    if out:
        return float(out)
    else:
        print(err)
        return


def get_segments(duration: float) -> typing.List[tuple]:
    """Get the start and end timestamp in seconds, of the AMIGOS video.

    Each of the videos were split into 20 second clips. For this, the first 20 seconds of each video, including 5
    seconds prior to the presentation of the stimuli, were extracted as first clip, then, starting from the 5s of
    the video (instant in which the stimuli started), n = ⌊(D)/(20s)⌋ non overlapping segments of 20s were extracted,
    with D being the duration of the stimuli video in seconds.
    Finally, the last 20 seconds of the video were extracted as final clip.
    eg. for an 85s clip
    #1: 0-20s
    #2: 5-25
    #3: seconds 20-40
    #4: seconds 40-60
    #5: seconds 60-80
    #6: seconds 65-85 (the last 20 seconds of the video)

    Args:
        duration (float): duration of a clip in seconds
    Returns:
        A list of tuples with the start and end second of each segment.
    """
    seg0: tuple = (0, 20)
    segments = [seg0]
    for step in range(5, int(duration), 20):
        if step+20 < duration:
            segments.append((step, step+20))
    segments.append((duration-20, duration))
    return segments


def seconds_to_strtime(seconds: float) -> str:
    """Formats a float representing seconds to str timestamp"""
    return str(datetime.timedelta(seconds=seconds))


def main(root_path, dst_dir, class_name):
    class_path = os.path.join(root_path, class_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir, class_name)
    if not os.path.exists(dst_class_path):
        os.makedirs(dst_class_path)

    for file_name in os.listdir(class_path):
        if '.mov' not in file_name:
            continue
        name, ext = os.path.splitext(file_name)
        video_len = probe_file(os.path.join(*[class_path, file_name]))
        if not video_len:
            continue
        segments = get_segments(video_len)
        video_file_path = os.path.join(class_path, file_name)
        for idx in range(len(segments)):
            dst_directory_path = os.path.join(*[dst_class_path, name, str(idx+1)])
            if not os.path.exists(dst_directory_path):
                os.makedirs(dst_directory_path)
            try:
                if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
                    subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                    print('remove {}'.format(dst_directory_path))
                    os.makedirs(dst_directory_path)
                else:
                    continue
            except Exception as e:
                print(dst_directory_path, e)
                continue
            segment_strt = str(segments[idx][0])
            segment_end = str(segments[idx][1])
            cmd = [
                'ffmpeg',
                '-to',
                segment_end,
                '-i',
                f'\"{video_file_path}\"',
                '-vf',
                f'trim={segment_strt}:{segment_end},setpts=PTS-STARTPTS,fps=25',
                '-q:v',
                '5',
                '-f',
                'image2',
                f'\"{dst_directory_path}/image_%05d.jpg\"'
            ]
            cmd = ' '.join(cmd)
            print(cmd)
            subprocess.call(cmd, shell=True)
            print('\n')


# main('Data', 'Frames', 'Exp1_P17_face')  # debug line
if __name__ == "__main__":
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    for root, dirs, files in os.walk(dir_path):
        for dirname in dirs:
            main(root, dst_dir_path, dirname)
