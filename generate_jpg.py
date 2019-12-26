from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
    rotation_range=5,  # 角度值，0~180，图像旋转
    width_shift_range=0.1,  # 水平平移，相对总宽度的比例
    height_shift_range=0.1,  # 垂直平移，相对总高度的比例
    shear_range=0.2,  # 随机错切换角度
    zoom_range=0.2,  # 随机缩放范围
    horizontal_flip=False,  # 一半图像水平翻转
    fill_mode='nearest'  # 填充新创建像素的方法
)


train_generator = datagen.flow_from_directory(
    r'C://Users/Administrator/Desktop/Atyun/mozhou/Atyuncalligraphy/data/huifengbudaowei',
# 'C://Users/Administrator/Desktop/Atyun/mozhou/Atyuncalligraphy/data/henghuajiangying/henghuajiangying_org', # 目标目录 r'C://Users/Administrator/Desktop/Workspace/python/calligraphy/point/left_gen'
    target_size=(400, 400), # 所有图像调整为150x150
    batch_size=5,
    shuffle=False,
    save_to_dir=r'C://Users/Administrator/Desktop/Atyun/mozhou/Atyuncalligraphy/data/huifengbudaowei/huifengbudaowei_gen',
    save_prefix='gen',
    save_format='jpg') # 二进制标签，我们用了binary_crossentropy损失函数


count = 0
for i in range(100):
    train_generator.next()
    count += 1
    if count % 10 == 0:
        print("Finish amount of :" + str(count) + "%")

    '''
    # 找到本地生成图，把9张图打印到同一张figure上

name_list = glob.glob(gen_path+'16/*')

fig = plt.figure()

for i in range(9):

img = Image.open(name_list[i])

sub_img = fig.add_subplot(331 + i)

sub_img.imshow(img)

plt.show()
    '''