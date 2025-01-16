# 공통메서드 모음
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 배열과 해상도 받아 이미지 배열 출력
def draw_images(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10

    fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)

    for i in range(rows):
        for j in range(cols):
            if i*10+j < n:
                axs[i, j].imshow(arr[i*10+j], cmap='gray_r')
                axs[i, j].axis('off')
    plt.show()

# 기본 선형그래프 그리기
def draw_plot(file_name, data_lists, x_label, y_lable, title_str, legend_lists):
    for i in range(0, len(data_lists)):
        plt.plot(data_lists[i])
    plt.xlabel(x_label)
    plt.ylabel(y_lable)
    plt.title(title_str)
    plt.legend(legend_lists)
    plt.savefig(file_name)



# 표만들기(객체생성)
def df_maker(column_list, index_list, contents):
    df = pd.DataFrame(contents, len(column_list), len(index_list))
    df.columns = column_list
    df.index = index_list
    return df
