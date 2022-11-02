import os
import shutil

def move_all_files():
    scenes = os.listdir(os.path.join('scene_datasets','mp3d'))
    for scene in scenes:
        if scene != 'v1':
            src = os.path.join('habitat_mp3d_connectivity_graphs',scene + '.json')
            dst = os.path.join('scene_datasets','mp3d',scene,scene + '.json')
            shutil.copyfile(src,dst)


if __name__ == '__main__':
    move_all_files()
