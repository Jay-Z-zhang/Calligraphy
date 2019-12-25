from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
    rotation_range=5,  # 角度值，0~180，图像旋转
    width_shift_range=0.2,  # 水平平移，相对总宽度的比例
    height_shift_range=0.2,  # 垂直平移，相对总高度的比例
    shear_range=0.2,  # 随机错切换角度
    zoom_range=0.2,  # 随机缩放范围
    horizontal_flip=False,  # 一半图像水平翻转
    fill_mode='nearest'  # 填充新创建像素的方法
)


train_generator = datagen.flow_from_directory(
    'C://Users/Administrator/Desktop/Workspace/python/calligraphy/point/left_str', # 目标目录 r'C://Users/Administrator/Desktop/Workspace/python/calligraphy/point/left_gen'
    target_size=(150, 150), # 所有图像调整为150x150
    batch_size=1,
    shuffle=False,
    save_to_dir=r'C://Users/Administrator/Desktop/Workspace/python/calligraphy/point/left_str/left_str_gen',
    save_prefix='left_str',
    save_format='jpg') # 二进制标签，我们用了binary_crossentropy损失函数

for i in range(1000):
    train_generator.next()