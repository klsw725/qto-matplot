import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
# import seaborn as sns

 

# import inputData as id

def open_file(file_name, *args):
    names = []
    for i in args:
        names.append(i)
    df = pd.read_csv(file_name,names=names)
    return df

def total_data(df,column):
    total = 0
    for i in range(0,len(df[column])):
        total = total + df[column][i]
    print(total)
    return total


#matplotlib 폰트설정
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 25
plt.rcParams['figure.figsize'] = (20,10)

color_dict = ['#0000FF','#FF0000']

plt.rcParams['font.size'] = 18

X = [800,1000,1200]

# ys, xs, patches = plt.hist(X, bins=3, range=(800,1400), weights=OLP_b, histtype='bar',rwidth=0.8)
# print(ys, xs, patches)
# plt.xticks([(xs[i]+xs[i+1])/2 for i in range(0, len(xs)-1)],
#            ["{}".format(xs[i]) for i in range(0, len(xs)-1)])
# plt.xlabel('전체 노드 수', fontproperties=fontprop, labelpad=10)
# plt.ylabel('전체 블랙 아웃 시간', fontproperties=fontprop, labelpad=10)


# df_o_total = []
# for i in id.df_o:
#     df_o_total.append(id.total_data(i,'blackout'))
# df_o = [np.mean(np.array(df_o_total[0:5])),np.mean(np.array(df_o_total[5:10])),np.mean(np.array(df_o_total[10:15]))]
#
# df_m_total = []
# for i in id.df_m:
#     df_m_total.append(id.total_data(i,'blackout'))
# df_m = [np.mean(np.array(df_m_total[0:5])),np.mean(np.array(df_m_total[5:10])),np.mean(np.array(df_m_total[10:15]))]
#
# B_data = np.array([[df_o[0] * 10 / 60,df_m[0] * 10 / 60],
#                   [df_o[1] * 10 / 60, df_m[1] * 10 / 60],
#                   [df_o[2] * 10 / 60, df_m[2] * 10 / 60]])
#
# diffrence_800 = abs(B_data[0][0] - B_data[0][1]) / B_data[0][0] * 100
# diffrence_1000 = abs(B_data[1][0] - B_data[1][1]) / B_data[1][0] * 100
# diffrence_1200 = abs(B_data[2][0] - B_data[2][1]) / B_data[2][0] * 100
# print(diffrence_800)
# print(diffrence_1000)
# print(diffrence_1200)
#
# df_B = pd.DataFrame(data=B_data ,index=[format(i,",") for i in X], columns=('LBDD','Proposed scheme'))
#
# fig1 = df_B.plot(kind='bar',color=[id.color_dict[0],id.color_dict[1]], edgecolor="black")
#
# plt.xlabel('Number of nodes', labelpad=10)
# plt.xticks(rotation=360)
# plt.ylabel('Total blackout time (Hours)', labelpad=10)
# plt.yticks([0,5000,10000,15000,20000,25000,30000,35000],[format(i,",") for i in [0,5000,10000,15000,20000,25000,30000,35000]])
#
# plt.subplots_adjust(left=0.15)
# # fig.set_size_inches(forward=True)
# ax = plt.gca()
# ax.tick_params(which='major', direction='in', length = 7)
# plt.savefig('totalblackout.png',bbox_inches='tight')
# plt.show()

# df_o_total = []
# for i in id.df_o:
#     df_o_total.append(id.total_data(i,'sink'))
# df_o = [np.mean(np.array(df_o_total[0:5])),np.mean(np.array(df_o_total[5:10])),np.mean(np.array(df_o_total[10:15]))]

# df_o = open_file('OLP600_10_5.csv',"index","blackout","sink", "canttransmit")
# df_o_total = total_data(df_o,'sink')

# df_m = open_file('MLP600_10_5.csv',"index","blackout","sink", "canttransmit")
# df_m_total = total_data(df_m,"sink")

# df_s = open_file('SLP600_10_5.csv',"index","blackout","sink", "canttransmit")
# df_s_total = total_data(df_s,"sink")

# df_c = open_file('CLP600_10_5.csv',"index","blackout","sink", "canttransmit")
# df_c_total = total_data(df_c,"sink")

# df_m_total = []
# for i in id.df_m:
#     df_m_total.append(id.total_data(i,'sink'))
# df_m = [np.mean(np.array(df_m_total[0:5])),np.mean(np.array(df_m_total[5:10])),np.mean(np.array(df_m_total[10:15]))]

S_data = np.array([[10131],[13280],[0],[15658],[2735.274],[8450]])
print(S_data)

# diffrence_800 = abs(S_data[0][0] - S_data[0][1]) / S_data[0][0] * 100
# diffrence_1000 = abs(S_data[1][0] - S_data[1][1]) / S_data[1][0] * 100
# diffrence_1200 = abs(S_data[2][0] - S_data[2][1]) / S_data[2][0] * 100
# print(diffrence_800)
# print(diffrence_1000)
# print(diffrence_1200)

# df_S = pd.DataFrame(data=S_data, index=('LBDD','Line Shift', 'LARCMS', 'Proposed scheme'), index=range(0,1))
index=['QTO','Full-offloading', 'Non-offloading', 'Q-Learning', 'DACO', 'HGOS']
# df_S = pd.DataFrame(data=S_data, index=tuple(index), columns=['totalblackoutdroptask'])
# print(df_S)
df = pd.DataFrame(data={"QTO":[1031], "Full-offloading":[13280], "Non-offloading":[0], "Q-Learning":[15658], "DACO":[2735.274], "HGOS":[8450]})

# fig2 = df_S.plot(kind='bar',color=[id.color_dict[0],id.color_dict[1]], edgecolor="black")
color_dict = ['blue','green','yellow','red','yellow','purple']
# colors = sns.color_palette('hls',len(index)) ## 색상 지정
# print(colors)
df.plot(kind='bar', edgecolor="black", color=color_dict, )
# plt.xlabel('Number of nodes', labelpad=10)
plt.ylabel('Amount of blackout drop task', labelpad=10)
plt.xticks(rotation=360)
# plt.yticks([i for i in range(0,int(max(S_data))+5000, 5000)],[format(i,",") for i in range(0,int(max(S_data))+5000, 5000)])

plt.subplots_adjust(left=0.15)

ax = plt.gca()
ax.tick_params(which='major', direction='in', length = 7)
plt.savefig('totalblackouttask.png',bbox_inches='tight')
plt.savefig('totalblackouttask.pdf',bbox_inches='tight')
plt.show()


