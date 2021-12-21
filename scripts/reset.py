import re
import shutil
from os import path, mkdir, listdir, walk
import uuid
from tqdm import tqdm
from typing import List


def move(source, target):
    if not path.exists(source):
        raise TypeError('source folder not exists !!!')
    if path.exists(target):
        raise TypeError('target folder exists !!!')

    mkdir(target)
    map = {dirt: 0 for dirt in listdir(source)}
    for dirt in listdir(source):
        dirt_path = path.join(target, dirt)
        if path.exists(dirt_path):
            raise TypeError(f'target children folder {dirt} exists !!!')

        mkdir(dirt_path)

        for file in listdir(dirt_path):
            if not file.endswith('.jpg') and not file.endswith('.jpeg') and not file.endswith('.png'):
                continue
            origin = path.join(target, dirt, file)
            dist = path.join(target, dirt, f'{dirt}_{map[dirt]}.jpg')
            shutil.move(origin, dist)
            map[dirt] += 1


def reorder(folder, prefix: str = '',  checkpoint=lambda x: True):
    """
    rename folder files with number

    folder: folder abs path
    prefix: prefix for each file to reorder
    checkpoint: check each file before reorder
    """
    if not path.exists(folder):
        raise TypeError('folder not exists')
    tmp_folder = f'/tmp/{uuid.uuid1()}'
    mkdir(tmp_folder)
    cnt = 0
    if prefix != '':
        prefix += '_'
    # src to tmp
    for file in listdir(folder):
        if not checkpoint(file):
            continue
        try:
            _, suffix = file.split('.')
            shutil.move(path.join(folder, file), path.join(
                tmp_folder, f'{prefix}{cnt}.{suffix}'))
            cnt += 1
        except Exception as e:
            print(e)
    # tmp to src
    for file in listdir(tmp_folder):
        if not checkpoint(file):
            continue
        try:
            _, suffix = file.split('.')
            shutil.move(path.join(tmp_folder, file), path.join(
                folder, file))
        except Exception as e:
            print(e)
    shutil.rmtree(tmp_folder)


def reset(source):
    if not path.exists(source):
        raise TypeError('source folder not exists !!!')
    for dirt in tqdm(listdir(source)):
        dirt_path = path.join(source, dirt)
        if not path.isdir(dirt_path):
            continue
        reorder(dirt_path, dirt, lambda x: x.endswith(
            '.jpg') or x.endswith('.jpeg') or x.endswith('.png'))


def merge(sources: List[str], target):
    """
    merge source folders and children folders to target folder
    """
    if not path.exists(target):
        mkdir(target)

    for source in tqdm(sources):
        root, dirts, files = next(walk(source))
        for file in files:
            src = path.join(root, file)
            dst = path.join(target, file)
            if path.exists(dst):
                try:
                    filename, suffix = file.split('.')
                    dst = path.join(target, f'{filename}_copy.{suffix}')
                except Exception as e:
                    print(e, file)
            shutil.copy(src, dst)
        for dirt in dirts:
            merge([path.join(root, dirt)], path.join(target, dirt))
        print(f'done for {source}')


if __name__ == '__main__':
    # source = '/home/djy/dataset/dataset1'
    # target = '/home/djy/dataset/dataset_1'
    # merge(['/home/djy/dataset/new_dataset',
    #        '/home/djy/dataset/new_dataset1'],
    #       '/home/djy/dataset/dataset1')
    reset('/home/djy/dataset/dataset1')
