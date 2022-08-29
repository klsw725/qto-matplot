# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import pandas as pd

# matplotlib 폰트설정
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 25
plt.rcParams['figure.figsize'] = (20, 10)

color_dict = ['#0000FF', '#FF0000']

plt.rcParams['font.size'] = 18

X = [5,10,15]
# 데이터
# age_category = ['10대 이하', '20대', '30대', '40대', '50대', '60대 이상'] ## 연령 카테고리, x축 눈금에 표시될 라벨
index = ['QTO', 'Full-offloading',
    'Non-offloading', 'Q-Learning', 'DACO', 'HGOS']
droptask = np.array([19,	0,	432000,	2,	29601,	0])
blackouttask = np.array([853762,	855180,	0,	848068,	156312,	403210])
forblock = np.array([0,0,0,0,0,0])
forslash = np.array([0,0,0,0,0,0])
# num_patient = [3,3,5,7,8,9] ## 방문환자수

# droptask = np.array([[9358,	0,	216000,	9386,	15600,	0],
#                     [ 18688, 	 0,	 432000,	 18973,	 29601,	 0],
#                     [28074,	0,	648000,	28158,	46800,	0]])

blackouttask1 = np.array([[70697.5,	427590,	49448,	226672.8,	151327.5,	201605],
                         [141554,	855180,	98901,	452480.8,	234468,	403210],
                         [212092.5,	1282770,	0,	680018.4,	453982.5,	604815]])

droptask = np.array([[9358,	18688, 28074],
                    [0 , 0 , 0],
                    [216000, 432000, 648000],
                    [9386, 18973, 28158],
                    [15600, 29601, 46800],
                    [0, 0, 0]])

blackouttask = blackouttask1.T
data1 = pd.DataFrame(data={'Because of latency':droptask, 'Because of blackout': blackouttask1}, index=index, columns=tuple(X))
# data2 = pd.DataFrame(data=blackouttask, index= index, columns=tuple(X))

# 시각화
# plt.figure(figsize=(10,10)) ## Figure 생성 사이즈는 10 by 10
colors = ['blue', 'green', 'orange', 'red', 'yellow', 'purple']
xtick_label_position = list(range(len(index)))  # x축 눈금 라벨이 표시될 x좌표
# plt.xticks(xtick_label_position, index, rotation=90)  # x축 눈금 라벨 출력

fig1 = data1.plot(kind='bar', color=colors, edgecolor="black", hatch='\\\\\\')
# fig2 = data2.plot(kind='bar', color=colors, edgecolor="black", stacked=True)


# plt.bar(xtick_label_position, droptask, color=colors, edgecolor="black")  # 바 차트 출력
# plt.bar(xtick_label_position, blackouttask, color=colors, bottom=droptask, hatch='\\\\\\', edgecolor="black")

# p1 = plt.bar(xtick_label_position, forblock, color='white', edgecolor="black", alpha=1)  # 바 차트 출력
# p2 = plt.bar(xtick_label_position, forslash, color='white', bottom=droptask, hatch='\\\\\\', edgecolor="black")

# ax2.bar(x, y, color='salmon', edgecolor='black', hatch='\\')
# plt.legend((p1[0], p2[0]), ('Because of blackout', 'Because of latency'))

# from matplotlib.legend import Legend
# leg = Legend(ax, lines[2:], ['line C', 'line D'],
#              loc='lower right', frameon=False)

# plt.title('지난달 연령별 방문환자 수',fontsize=20) ## 타이틀 출력
# plt.xlabel('연령') ## x축 라벨 출력
# leg = plt.legend().remove()

# ax.get_legend().remove()
# plt.ylim((0, 120000))

plt.yticks([i for i in range(0, 1200000, 200000)], [format(i, ",") for i in range(0, 1200000, 200000)])
# plt.yticks([0,200000, 400000, 600000, 800000, 1000000, 1200000], ["0","200,000", "400,000" "600,000" "800,000" "1,000,000", "1,200,000"])
plt.xticks(rotation=360)
plt.ylabel('Total drop task', labelpad=10)  # y축 라벨 출력
plt.show()
plt.savefig('droptask.png', bbox_inches='tight')
plt.savefig('droptask.pdf', bbox_inches='tight')
