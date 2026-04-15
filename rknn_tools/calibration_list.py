import os

img_dir = 'E:/bird/dataset/Calibration_data_1200'
output_file = './dataset1200.txt'

extensions = ('.jpg', '.jpeg', '.png', '.bmp')

img_paths = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith(extensions)
]

img_paths.sort()

with open(output_file, 'w') as f:
    for filename in img_paths:
        path = './{}/{}'.format(os.path.basename(img_dir), filename)
        f.write(path + '\n')

print(f'共写入 {len(img_paths)} 张图片路径到 {output_file}')
