import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def degrade_image(img, blur_kernel_size, blur_angle, noise_std):
    # 进行运动模糊
    motion_kernel = np.zeros((blur_kernel_size, blur_kernel_size))
    motion_kernel[int((blur_kernel_size-1)/2),:] = np.ones(blur_kernel_size)
    motion_kernel = cv2.warpAffine(motion_kernel, cv2.getRotationMatrix2D((blur_kernel_size/2-0.5,blur_kernel_size/2-0.5), blur_angle, 1.0), (blur_kernel_size, blur_kernel_size))
    motion_kernel = motion_kernel / np.sum(motion_kernel)
    img_blur = cv2.filter2D(img, -1, motion_kernel)
    
    # 添加高斯噪声
    noise = np.random.normal(0, noise_std, img.shape)
    img_noise = img_blur + noise
    
    return img_noise, motion_kernel

def restore_image(img_degraded, blur_kernel, reg_param):
    # 将图像和退化核转换为频率域
    F = np.fft.fft2(img_degraded)
    H = np.fft.fft2(blur_kernel, s=img_degraded.shape[:2], axes=(0, 1))
    Hconj = np.conj(H)
    
    # 计算最优的 H(u,v)
    A = np.zeros((img_degraded.size, img_degraded.size))
    b = np.zeros(img_degraded.size)
    i = 0
    for u in range(img_degraded.shape[0]):
        for v in range(img_degraded.shape[1]):
            Huv = H[u,v]
            Hconj_uv = Hconj[u,v]
            Fuv = F[u,v]

            # print out the dimensions of the matrices for debugging
            print('Huv.shape =', Huv.shape)
            print('Hconj_uv.shape =', Hconj_uv.shape)
            print('Fuv.shape =', Fuv.shape)
            print('A[i,:].shape =', A[i,:].shape)
            
            A[i,:] = np.concatenate((np.ravel(np.real(Hconj_uv)), np.ravel(np.imag(Hconj_uv))))
            b[i] = np.real(Hconj_uv * Fuv)
            i += 1
    reg = np.eye(img_degraded.size) * reg_param
    h = np.linalg.solve((A.T @ A + reg), A.T @ b)
    Hest = np.zeros(img_degraded.shape[:2], dtype=np.complex128)
    Hest.real = h[:img_degraded.size//2].reshape(img_degraded.shape[:2])
    Hest.imag = h[img_degraded.size//2:].reshape(img_degraded.shape[:2])
    
    # 恢复图像
    Fest = F / Hest
    img_restored = np.fft.ifft2(Fest).real
    
    # 将图像限制在 0-255 范围内
    img_restored = np.clip(img_restored, 0, 255)
    img_restored = img_restored.astype(np.uint8)
    
    return img_restored


# 读取测试图像
img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 对测试图像进行退化
blur_kernel_size = 15
blur_angle = 45
noise_std = 20
img_degraded, blur_kernel = degrade_image(img, blur_kernel_size, blur_angle, noise_std)

# 恢复图像
reg_param = 1e-3
img_restored = restore_image(img_degraded, blur_kernel, reg_param)

# 显示结果
plt.subplot(121)
plt.imshow(img_degraded)
plt.title('Degraded Image')

plt.subplot(122)
plt.imshow(img_restored)
plt.title('Restored Image')

plt.show()
