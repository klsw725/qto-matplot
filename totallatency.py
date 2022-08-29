# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import numpy as np

# matplotlib 폰트설정
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 25
plt.rcParams['figure.figsize'] = (20,10)

color_dict = ['#0000FF','#FF0000']

plt.rcParams['font.size'] = 22

X = ["5 devices", "10 devices", "15 devices"]
## 데이터
# age_category = ['10대 이하', '20대', '30대', '40대', '50대', '60대 이상'] ## 연령 카테고리, x축 눈금에 표시될 라벨
index=['QTO','Full-offloading', 'Non-offloading', 'Q-Learning', 'DACO', 'HGOS']
# data1 = np.array([[1.247774883,	0.0603,	3,	1.382899803,	1.631764353	,2.006532],
#                 [1.246117237,	0.0603,	3,	1.385567003,	1.49996917,	2.006532],
#                 [1.247774883,	0.0603,	3,	1.382899803,	1.631764353,	2.006532]])

# num_patient = [3,3,5,7,8,9] ## 방문환자수

def plot_clustered_stacked(dfall, labels=None, title="",  H="///", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      edgecolor="black",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      color='white',
                      **kwargs)  # make bar plots

    k=0                      
    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                # rect.set_color(colors[int(i/n_col)])
                rect.set_width(1 / float(n_df + 1))
                rect.set_linewidth(0.5)

                if k > 11 : 
                    rect.set_hatch("\\\\\\") #edited part 
                k+=1

    # axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)
    

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        # n.append(axe.bar(0, 0, color="#3B75AF", hatch=H * i))
        n.append(axe.bar(0, 0, color="white", hatch=H * i, edgecolor="black"))
    n[-1] = axe.bar(0, 0, color="white", hatch='\\\\\\', edgecolor="black")

    # l1 = axe.legend(h[:n_col], l[:n_col], loc=[0.74, 0.87 ])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[0.78, 0.77]) 
    # axe.add_artist(l1)
    return axe


data1 = np.array([[1.247774883*1000],
                  [0.0603*1000],
                  [3*1000],
                  [1.382899803*1000],
                  [1.631764353*1000],
                  [2.006532*1000]])
data2 = np.array([[1.246117237*1000],
                  [0.0603*1000],
                  [3*1000],
                  [1.382899803*1000],
                  [1.49996917*1000],
                  [2.006532*1000]])
data3 = np.array([[1.247774883*1000],
                  [0.0603*1000],
                  [3*1000],
                  [1.382899803*1000],
                  [1.631764353*1000],
                  [2.006532*1000]])
# blackouttask = blackouttask1.T

data11 = pd.DataFrame(data=data1, index=index)
data22 = pd.DataFrame(data=data2, index= index)
data33 = pd.DataFrame(data=data3, index= index)

## 시각화
# plt.figure(figsize=(10,10)) ## Figure 생성 사이즈는 10 by 10
# colors = ['blue','green','yellow','red','yellow','purple']
# xtick_label_position = list(range(len(index))) ## x축 눈금 라벨이 표시될 x좌표
plot_clustered_stacked([data11, data22, data33],X)
# data_S.plot.xticks(xtick_label_position, index) ## x축 눈금 라벨 출력
 
# fig1 = data_S.plot(kind='bar', hatch edgecolor="black") ## 바 차트 출력
plt.xticks(rotation=360)
 
# plt.title('지난달 연령별 방문환자 수',fontsize=20) ## 타이틀 출력
# plt.xlabel('연령') ## x축 라벨 출력
# leg = plt.legend().remove()

# ax.get_legend().remove()

plt.ylabel('Latency per task(ms)', labelpad=10) ## y축 라벨 출력
plt.show()
plt.savefig('latencypertask.png',bbox_inches='tight')
plt.savefig('latencypertask.pdf',bbox_inches='tight')

