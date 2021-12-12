import cv2
import os


def frames_extraction(source_path: str, target_path: str, frames_frequency=1) -> None:
    """
    固定间隔提取视频关键帧
    :param source_path:视频源文件地址
    :param target_path:关键帧输出地址
    :return:None
    """
    # output directory path
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # 创建视频对象
    cap = cv2.VideoCapture(source_path)
    print(f'Frame rate: {cap.get(cv2.CAP_PROP_FPS)}')
    print(
        f'Number of frames in the video file: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')

    cnt, times = 0, 0
    while True:

        # 读取视频帧
        ret, image = cap.read()

        if not ret:
            break

        # 指定帧率保存帧图片
        if times % frames_frequency == 0:
            cv2.imwrite(f'{target_path}/{cnt}_{times}.jpg', image)
            print(f'{cnt}_{times}.jpg')
            cnt += 1

        times += 1

    print('frames extraction finished')
    # 释放
    cap.release()


if __name__ == '__main__':
    # 打开当前文件夹下的 video.mp4 视频
    # for i in range(5):
    folder = '/home/djy/Videos'
    source = os.path.join(folder, 'video2.mov')
    target = os.path.join(
        folder, f'frames_video2')
    print(source, target)
    frames_extraction(source, target)
