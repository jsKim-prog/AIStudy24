# 파일 읽어오기
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from util import *
data = np.load('fruits_300.npy')
print(f"원본데이터 정보 : {data.shape}")   # 300장, 100*100(해상도)
# 데이터 확인
#plt.imshow(data[0], cmap=cm.gray_r)
#plt.show()
# fig, axs = plt.subplots(1, 3)
# axs[0].imshow(data[0], cmap='gray_r')
# axs[1].imshow(data[100], cmap='gray_r')
# axs[2].imshow(data[200], cmap='gray_r')
# plt.show()

# 데이터 처리(3차원->2차원 배열로 변경)
print(data[0])
fruits = data.reshape(-1, 100*100)
print(fruits.shape)
print(fruits[:2])