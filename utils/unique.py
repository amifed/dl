from PIL import Image
from shutil import move
import imagehash
import os

dist = '/home/djy/dataset/uni_dataset'

classes = ('bzx', 'cwx', 'hdx', 'mtx', 'nqx', 'nzx', 'qtx', 'sjx', 'zxx')
mp = {cls: 0 for cls in classes}
uni_mp = {cls: 0 for cls in classes}

for cls in classes:
    st = set()
    folder = os.path.join(dist, cls)
    mp[cls] = len(os.listdir(folder))
    for file in os.listdir(folder):
        if not file.endswith('.jpg'):
            continue
        # image = Image.open(os.path.join(folder, file))
        # hh = imagehash.dhash(image)
        # if hh in st:
        #     continue
        # st.add(hh)
        source = os.path.join(folder, file)
        target = os.path.join(dist, cls, f'{cls}_{uni_mp[cls]}.jpg')
        move(source, target)
        uni_mp[cls] += 1

    print(f'{cls}: cnt: {mp[cls]}, uni cnt: {len(st)}')

print(mp)
print(uni_mp)
