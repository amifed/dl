from shutil import move
import os

origin = '/home/djy/dataset/uni_dataset'
dist = '/home/djy/dataset/uni_dataset_1'

classes = ('bzx', 'cwx', 'hdx', 'mtx', 'nqx', 'nzx', 'qtx', 'sjx', 'zxx')
mp = {cls: 0 for cls in classes}

if (not os.path.exists):
    os.mkdir(dist)

for cls in classes:
    folder = os.path.join(origin, cls)
    if not os.path.exists(folder):
        continue
    for file in os.listdir(folder):
        if not file.endswith('.jpg') and not file.endswith('.jpeg'):
            continue
        source = os.path.join(folder, file)
        target_cls = os.path.join(dist, cls)
        if (not os.path.exists(target_cls)):
            os.mkdir(target_cls)
        target = os.path.join(dist, cls, f'{cls}_{mp[cls]}.jpg')
        move(source, target)
        mp[cls] += 1

    print(f'{cls}: cnt: {mp[cls]}')

print(mp)
