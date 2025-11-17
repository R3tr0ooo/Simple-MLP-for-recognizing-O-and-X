import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from scipy import ndimage
import cv2



#能不能看懂靠缘分嗷，扭曲啥的涉及计算机视觉，让Deepseek写的






# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def apply_distortion(image, distortion_type=None):
    """
    应用各种扭曲效果到图像
    """
    if distortion_type is None:
        # 随机选择一种扭曲类型
        distortion_type = np.random.choice([
            'rotation', 'scaling', 'translation', 'elastic', 'skew', 
            'noise', 'blur', 'perspective', 'none'
        ], p=[0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    
    h, w = image.shape
    
    if distortion_type == 'rotation':
        # 随机旋转 (-15° 到 15°)
        angle = np.random.uniform(-90, 90)
        return ndimage.rotate(image, angle, reshape=False, mode='constant', cval=0)
    
    elif distortion_type == 'scaling':
        # 随机缩放 (0.8 到 1.2)
        scale = np.random.uniform(0.8, 1.2)
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (new_w, new_h))
        
        # 如果缩放后尺寸不同，填充或裁剪到原始尺寸
        if scale > 1:
            # 裁剪中心部分
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return scaled[start_h:start_h+h, start_w:start_w+w]
        else:
            # 填充到原始尺寸
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            padded = np.zeros((h, w))
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled
            return padded
    
    elif distortion_type == 'translation':
        # 随机平移 (-3 到 3 像素)
        tx = np.random.randint(-3, 4)
        ty = np.random.randint(-3, 4)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    elif distortion_type == 'elastic':
        # 弹性变形
        alpha = w * 0.8
        sigma = w * 0.15
        random_state = np.random.RandomState(None)
        
        dx = ndimage.gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = ndimage.gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        return ndimage.map_coordinates(image, indices, order=1, mode='constant').reshape(h, w)
    
    elif distortion_type == 'skew':
        # 倾斜变形
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # 随机生成倾斜点
        skew_x = np.random.uniform(-0.1, 0.1) * w
        skew_y = np.random.uniform(-0.1, 0.1) * h
        
        pts2 = np.float32([
            [skew_x, 0],
            [w - skew_x, 0],
            [0, h - skew_y],
            [w, h - skew_y]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    elif distortion_type == 'noise':
        # 添加高斯噪声
        noise = np.random.normal(0, 0.1, image.shape)
        return np.clip(image + noise, 0, 1)
    
    elif distortion_type == 'blur':
        # 高斯模糊
        kernel_size = np.random.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif distortion_type == 'perspective':
        # 透视变换
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # 随机生成透视点
        perspective = np.random.uniform(-0.15, 0.15) * w
        
        pts2 = np.float32([
            [perspective, perspective],
            [w - perspective, perspective],
            [perspective, h - perspective],
            [w - perspective, h - perspective]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    else:  # 'none'
        return image

def generate_o_pattern():
    """生成O的图案 - 圆形"""
    img = np.zeros((20, 20))
    center = (9.5, 9.5)
    for i in range(20):
        for j in range(20):
            dist = np.sqrt((i-center[0])**2 + (j-center[1])**2)
            if 5 <= dist <= 7:
                img[i,j] = 1
    
    # 应用随机扭曲
    img = apply_distortion(img)
    
    return img + np.random.normal(0, 0.05, (20, 20))

def generate_x_pattern():
    """生成X的图案 - 两条交叉线"""
    img = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            if abs(i - j) <= 1.5 or abs(i + j - 19) <= 1.5:
                img[i,j] = 1
    
    # 应用随机扭曲
    img = apply_distortion(img)
    
    return img + np.random.normal(0, 0.05, (20, 20))

def generate_data(num_samples=2000):
    """生成训练数据"""
    X_data = []
    y_data = []
    
    for i in range(num_samples):
        if np.random.random() > 0.5:
            img = generate_o_pattern()
            label = [1, 0]  # O
        else:
            img = generate_x_pattern()
            label = [0, 1]  # X
            
        X_data.append(img)
        y_data.append(label)
    
    return np.array(X_data), np.array(y_data)

# 可视化扭曲效果
def visualize_distortions():
    """可视化各种扭曲效果"""
    # 生成原始O和X
    original_o = np.zeros((20, 20))
    original_x = np.zeros((20, 20))
    
    center = (9.5, 9.5)
    for i in range(20):
        for j in range(20):
            dist = np.sqrt((i-center[0])**2 + (j-center[1])**2)
            if 5 <= dist <= 7:
                original_o[i,j] = 1
            
            if abs(i - j) <= 1.5 or abs(i + j - 19) <= 1.5:
                original_x[i,j] = 1
    
    # 应用各种扭曲
    distortions = [
        ('原始', 'none'),
        ('旋转', 'rotation'),
        ('缩放', 'scaling'),
        ('平移', 'translation'),
        ('弹性变形', 'elastic'),
        ('倾斜', 'skew'),
        ('噪声', 'noise'),
        ('模糊', 'blur'),
        ('透视', 'perspective')
    ]
    
    fig, axes = plt.subplots(2, len(distortions), figsize=(18, 6))
    fig.suptitle('X 和 O 的各种扭曲效果', fontsize=20, fontweight='bold')
    
    for i, (name, dist_type) in enumerate(distortions):
        # O的扭曲
        o_distorted = apply_distortion(original_o.copy(), dist_type)
        axes[0, i].imshow(o_distorted, cmap='viridis')
        axes[0, i].set_title(f'O - {name}', fontsize=12)
        axes[0, i].axis('off')
        
        # X的扭曲
        x_distorted = apply_distortion(original_x.copy(), dist_type)
        axes[1, i].imshow(x_distorted, cmap='viridis')
        axes[1, i].set_title(f'X - {name}', fontsize=12)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 生成数据
print("生成带扭曲的训练数据...")
X, y = generate_data(2000)
X = X.astype('float32')
X = (X - X.min()) / (X.max() - X.min())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")

# 创建单隐藏层模型
model = Sequential([
    Flatten(input_shape=(20, 20)),
    Dense(128, activation='relu', kernel_regularizer='l2'),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 添加早停
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 训练模型
print("开始训练模型...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=25,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n测试准确率: {test_acc:.4f}")

# 绘制训练历史
plt.figure(figsize=(14, 6))

# 准确率图表
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率', linewidth=2)
plt.plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
plt.title('模型训练准确率变化', fontsize=14, fontweight='bold')
plt.xlabel('训练轮次 (Epochs)', fontsize=12)
plt.ylabel('准确率 (Accuracy)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(0, len(history.history['accuracy'])-1)

# 损失图表
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失', linewidth=2)
plt.plot(history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('模型训练损失变化', fontsize=14, fontweight='bold')
plt.xlabel('训练轮次 (Epochs)', fontsize=12)
plt.ylabel('损失值 (Loss)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(0, len(history.history['loss'])-1)

plt.tight_layout()
plt.show()

# 可视化扭曲效果
print("可视化各种扭曲效果...")
visualize_distortions()

# 预测样本图形输出功能
def plot_prediction_samples_with_distortion(model, num_samples=12):
    """
    生成并显示带扭曲的预测样本图形
    """
    # 创建图形
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    fig.suptitle('带扭曲的神经网络预测样本示例', fontsize=20, fontweight='bold', y=0.98)
    
    # 确保axes是二维数组
    axes = axes.flat
    
    distortion_types = ['rotation', 'scaling', 'translation', 'elastic', 'skew', 'noise', 'blur', 'perspective']
    
    for i in range(num_samples):
        # 随机生成测试样本
        if np.random.random() > 0.5:
            test_img = generate_o_pattern()
            true_label = "O"
            true_class = 0
        else:
            test_img = generate_x_pattern()
            true_label = "X"
            true_class = 1
        
        # 预处理图像
        test_img_processed = test_img.reshape(1, 20, 20).astype('float32')
        test_img_processed = (test_img_processed - test_img_processed.min()) / (test_img_processed.max() - test_img_processed.min())
        
        # 进行预测
        prediction = model.predict(test_img_processed, verbose=0)
        predicted_class = np.argmax(prediction[0])
        predicted_label = "O" if predicted_class == 0 else "X"
        confidence = max(prediction[0])
        
        # 确定颜色（正确预测为绿色，错误预测为红色）
        color = 'green' if predicted_label == true_label else 'red'
        
        # 绘制图像
        ax = axes[i]
        ax.imshow(test_img, cmap='viridis')
        
        # 获取应用的扭曲类型（通过比较原始和扭曲后的图像）
        distortion_name = "随机扭曲"
        
        ax.set_title(f'样本 {i+1}\n真实: {true_label} | 预测: {predicted_label}\n置信度: {confidence:.4f}', 
                    color=color, fontsize=12, pad=10)
        ax.set_xlabel(f'O概率: {prediction[0][0]:.4f} | X概率: {prediction[0][1]:.4f}', 
                     fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加边框颜色表示预测正确与否
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    # 打印统计信息
    print(f"\n预测样本统计:")
    print(f"总样本数: {num_samples}")
    print(f"测试准确率: {test_acc:.4f}")

# 比较有扭曲和无扭曲模型的性能
def compare_with_without_distortion():
    """
    比较使用扭曲数据和不使用扭曲数据的模型性能
    """
    print("\n" + "="*60)
    print("比较有扭曲和无扭曲数据训练的模型性能")
    print("="*60)
    
    # 生成无扭曲的数据
    def generate_without_distortion():
        X_data = []
        y_data = []
        
        for i in range(1000):
            if np.random.random() > 0.5:
                # 生成O的图案 - 无扭曲
                img = np.zeros((20, 20))
                center = (9.5, 9.5)
                for i_ in range(20):
                    for j in range(20):
                        dist = np.sqrt((i_-center[0])**2 + (j-center[1])**2)
                        if 5 <= dist <= 7:
                            img[i_,j] = 1
                label = [1, 0]  # O
            else:
                # 生成X的图案 - 无扭曲
                img = np.zeros((20, 20))
                for i_ in range(20):
                    for j in range(20):
                        if abs(i_ - j) <= 1.5 or abs(i_ + j - 19) <= 1.5:
                            img[i_,j] = 1
                label = [0, 1]  # X
                
            X_data.append(img)
            y_data.append(label)
        
        return np.array(X_data), np.array(y_data)
    
    # 生成无扭曲数据
    X_no_dist, y_no_dist = generate_without_distortion()
    X_no_dist = X_no_dist.astype('float32')
    X_no_dist = (X_no_dist - X_no_dist.min()) / (X_no_dist.max() - X_no_dist.min())
    
    X_train_nd, X_test_nd, y_train_nd, y_test_nd = train_test_split(
        X_no_dist, y_no_dist, test_size=0.2, random_state=42, stratify=y_no_dist
    )
    
    # 创建无扭曲模型
    model_no_dist = Sequential([
        Flatten(input_shape=(20, 20)),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    
    model_no_dist.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练无扭曲模型
    print("训练无扭曲数据模型...")
    history_nd = model_no_dist.fit(
        X_train_nd, y_train_nd,
        batch_size=32,
        epochs=50,
        validation_data=(X_test_nd, y_test_nd),
        verbose=0
    )
    
    # 评估两个模型
    test_loss_nd, test_acc_nd = model_no_dist.evaluate(X_test_nd, y_test_nd, verbose=0)
    
    print(f"\n模型性能比较:")
    print(f"使用扭曲数据的模型准确率: {test_acc:.4f}")
    print(f"不使用扭曲数据的模型准确率: {test_acc_nd:.4f}")
    print(f"性能提升: {test_acc - test_acc_nd:.4f}")
    
    # 在扭曲数据上测试两个模型
    dist_loss, dist_acc = model_no_dist.evaluate(X_test, y_test, verbose=0)
    print(f"\n在扭曲数据上的泛化能力:")
    print(f"使用扭曲数据的模型: {test_acc:.4f}")
    print(f"不使用扭曲数据的模型: {dist_acc:.4f}")
    
    # 绘制比较图
    plt.figure(figsize=(10, 6))
    models = ['使用扭曲数据', '不使用扭曲数据']
    accuracies = [test_acc, test_acc_nd]
    colors = ['lightgreen', 'lightcoral']
    
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black', alpha=0.7)
    plt.title('模型性能比较', fontsize=16, fontweight='bold')
    plt.ylabel('测试准确率', fontsize=12)
    plt.ylim(0, 1)
    
    # 在柱状图上添加数值
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# 执行可视化功能
print("=" * 60)
print("带扭曲数据的神经网络 O/X 识别屎山")
print("=" * 60)

# 1. 显示带扭曲的预测样本
print("\n1. 带扭曲的多样本预测结果:")
plot_prediction_samples_with_distortion(model, num_samples=12)
'''
# 2. 比较有扭曲和无扭曲模型
compare_with_without_distortion()

# 3. 显示混淆矩阵
print("\n2. 模型性能评估:")
def plot_simple_confusion_matrix(model, num_test_samples=200):
    """
    绘制简化的混淆矩阵
    """
    # 生成测试样本
    y_true = []
    y_pred = []
    
    for i in range(num_test_samples):
        if np.random.random() > 0.5:
            test_img = generate_o_pattern()
            true_label = 0  # O
        else:
            test_img = generate_x_pattern()
            true_label = 1  # X
        
        # 预处理和预测
        test_img_processed = test_img.reshape(1, 20, 20).astype('float32')
        test_img_processed = (test_img_processed - test_img_processed.min()) / (test_img_processed.max() - test_img_processed.min())
        prediction = model.predict(test_img_processed, verbose=0)
        predicted_label = np.argmax(prediction[0])
        
        y_true.append(true_label)
        y_pred.append(predicted_label)
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵 (带扭曲数据)', fontsize=16, fontweight='bold')
    plt.colorbar()
    
    classes = ['O', 'X']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 打印准确率
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\n混淆矩阵分析 (基于 {num_test_samples} 个样本):")
    print(f"总体准确率: {accuracy:.4f}")
    print(f"O类准确率: {cm[0,0] / np.sum(cm[0,:]):.4f}" if np.sum(cm[0,:]) > 0 else "N/A")
    print(f"X类准确率: {cm[1,1] / np.sum(cm[1,:]):.4f}" if np.sum(cm[1,:]) > 0 else "N/A")

plot_simple_confusion_matrix(model)

print("\n训练完成！模型现在可以识别带各种扭曲的O和X图案。")
'''