# ==================== 导入库 ====================
# numpy: 数值计算库，提供高性能数组ndarray，是pandas和sklearn的底层运算引擎
import numpy as np
# pandas: 数据分析库，DataFrame像一张Excel表格，是数据清洗和特征工程的核心工具
import pandas as pd
# matplotlib.pyplot: 画图工具箱，本文件用它画热力图、回归曲线、系数图等
import matplotlib.pyplot as plt

# ---- scikit-learn 的工具 ----
# GroupKFold: 分组K折交叉验证，同一个host_id（房东）的房源不会被拆到训练/测试两边
# train_test_split: 把数据随机切成训练集和测试集
# cross_validate: 一行代码完成K折交叉验证，返回每一折的分数
from sklearn.model_selection import GroupKFold, train_test_split, cross_validate
# Pipeline: 流水线，把"预处理步骤 + 模型"串成一个整体，fit/predict一步到位
from sklearn.pipeline import Pipeline
# ColumnTransformer: 按列分别做预处理（数值列原样通过、文本列独热编码）
from sklearn.compose import ColumnTransformer
# OneHotEncoder: 独热编码，把"房子类型、房间类型"这类文本列变成0/1列
from sklearn.preprocessing import OneHotEncoder
# LinearRegression: 普通最小二乘线性回归，y = w1*x1 + w2*x2 + ... + b
from sklearn.linear_model import LinearRegression
# root_mean_squared_error: 均方根误差（RMSE），衡量预测值和真实值的平均偏差，单位与y相同
from sklearn.metrics import root_mean_squared_error
# os: 操作系统接口，这里用来创建保存图表的figures文件夹
import os
# StandardScaler: 标准化处理器，把每列数值变成 均值=0、标准差=1 的分布
from sklearn.preprocessing import StandardScaler
# Ridge/RidgeCV: 岭回归（L2惩罚，把系数整体压小但不压成0）及其自动选λ的版本
# Lasso/LassoCV: 套索回归（L1惩罚，把不重要的系数直接压成0）及其自动选λ的版本
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV


# ==================== 第0步：全局设置（中文图表字体 + 提前准备保存目录） ====================
# matplotlib画图时指定中文字体：
# Windows自带"微软雅黑/黑体"，能渲染中文；不指定的话图上所有中文会变成方框（乱码）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
# 坐标轴上的负号用普通减号显示，避免负号变成方框
plt.rcParams['axes.unicode_minus'] = False

# 提前创建保存图表的figures文件夹（必须在任何plt.savefig之前执行，
# 否则第一次保存时目录不存在会直接报错）
os.makedirs('figures', exist_ok=True)


# ==================== 第1步：加载数据 ====================
# 数据是雅典Airbnb房源快照：14337个房源、90列，每行一个房源
df = pd.read_csv('data/listings.csv')

# ==================== 第2步：检查数据规模是否达标 ====================
# 项目Brief要求：房源数量 >= 5000，且要有数百个同时挂多套房的房东
print(f"房源总数: {len(df)}")
print("  说明：项目Brief要求样本量>=5000，这个数就是检验的第一条硬指标")
# df["host_id"].value_counts() 统计每个房东挂了几套房
host_counts = df["host_id"].value_counts()
# (host_counts > 1).sum() 统计"拥有2套及以上"的房东有多少个
print(f"拥有多套房的房东数: {(host_counts > 1).sum()}")
print("  说明：项目要求有'数百个'这样的房东。重复房东够多，"
      "'按房东分组'的交叉验证才有意义（模型才不是靠认出房东作弊）")

# ==================== 第3步：安全地清洗价格列 ====================
# 原始price列是文本，里面混着 $、€、逗号等脏字符，无法直接当数字用
# pd.to_numeric(...) 把字符串尽力转成浮点数
#   .str.replace(r'[^\d.]', '', regex=True)：只保留数字0-9和小数点，其余全删
#   errors='coerce'：转不了的值（如空字符串）变成NaN而不是报错，程序不崩
df['price_clean'] = pd.to_numeric(
    df['price']
    .astype(str)
    .str.replace(r'[^\d.]', '', regex=True),
    errors='coerce'
)

# ==================== 第4步：筛选出雅典的有效房源 ====================
# exclusion_rule 是布尔掩码（True/False序列），三条条件用&（按位与）连接，必须同时满足：
#   1. 价格 > 0          —— 排除0元房源（可能是缺失或异常）
#   2. 价格 <= 1000      —— 排除天价房源，上限设为1000欧元
#   3. 地区 == 'Athens, Greece' —— 只分析雅典的房源（数据里其实还有其他城市）
exclusion_rule = (df['price_clean'] > 0) & (df['price_clean'] <= 1000) & (df['host_location'] == 'Athens, Greece')
# df[布尔掩码] 只保留掩码为True的行，.copy()生成独立副本，避免污染原df
df_filtered = df[exclusion_rule].copy()

# ==================== 第5步：计算"什么都不做"的基线 ====================
# 机器学习铁律：模型必须比"瞎猜均值"强，否则模型没有价值
# .mean() 价格平均值；.median() 中位数（更抗极端值干扰）
mean_baseline = df_filtered['price_clean'].mean()
median_baseline = df_filtered['price_clean'].median()
print(f"筛选后房源总数: {len(df_filtered)}")
print("  说明：去掉非雅典、0元/超1000欧的房源后，真正用于建模的样本量")
print(f"清洗后均值基线: €{mean_baseline:.2f}")
print("  说明：后面每个模型都会拿自己的误差和'直接猜均价'比，比不过就没价值")
print(f"中位价格: €{median_baseline:.2f}")
print("  说明：一半房源低于这个价，比均值更能代表'雅典的典型房价'")
print(f"筛选后最高价: €{df_filtered['price_clean'].max():.2f}")
print("  说明：价格上限——因为上面筛选时卡了 <=1000 欧元")


# ==================== 第6步：第一版特征集合（仅用来展示分组折）====================
# 注意：下面的 categorical_features / numeric_features 在后面的第7步会被正式版本覆盖
# 这里先定义一版简化的特征列表，只是为了展示"分组K折"把数据切成了几份
categorical_features = [
    'property_type',   # 房屋类型（文本，暂未使用）
    'room_type'        # 出租方式（文本，暂未使用）
]
numeric_features = [
    'hosts_time_as_user_months',   # 房东注册为Airbnb用户的月数
    'hosts_time_as_host_months',   # 房东首次出租至今的月数（经验值）
    'host_is_superhost',           # 是否超级房东
    'host_has_profile_pic',        # 是否有头像
    'host_identity_verified',      # 是否通过身份验证
    'accommodates',                # 可容纳人数
    'bathrooms',                   # 浴室数量
    'bedrooms',                    # 卧室数量
    'beds',                        # 床的数量
    'minimum_nights',              # 最少入住天数
    'maximum_nights',              # 最多入住天数
    'host_listings_count',         # 房东挂的房源总数
    'review_scores_rating',        # 综合评分（0-100）
    'reviews_per_month'            # 每月评价数
]
# dropna(subset=...) 只丢弃这些列里有缺失的行，得到一份无缺失的回归数据
df_filtered1 = df_filtered.dropna(subset=numeric_features)
# X 是特征矩阵，y 是目标变量（要预测的价格）
X = df_filtered1[['hosts_time_as_user_months', 'hosts_time_as_host_months', 'host_is_superhost', 'host_has_profile_pic',
                  'host_identity_verified', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms',
                  'beds', 'minimum_nights', 'maximum_nights', 'review_scores_rating', 'host_listings_count']]
y = df_filtered1['price_clean']
# 分组K折：同一房东的多套房不会被拆到训练/测试两边（避免靠"认出房东"作弊）
gk5 = GroupKFold(n_splits=5)
groups = df_filtered1['host_id']

# 遍历每一折，打印训练/测试集大小（这步只是为了确认划分器正常工作）
for fold, (train_idx, test_idx) in enumerate(gk5.split(X, y, groups)):
    print(f"第{fold + 1}折")
    print(f"训练房源数: {len(train_idx)}")
    print(f"测试房源数: {len(test_idx)}")
print("说明：这只是确认分组K折把数据切成了5份；同一房东的房源始终在同一折里，"
      "训练集和测试集没有重叠房东")


# ==================== 第7步：正式的特征集合 + 布尔列转0/1 ====================
# 下面的定义会覆盖第6步的版本——这才是正式用于建模的特征集合
categorical_features = [
    'property_type',             # 房屋类型（文本）
    'room_type',                 # 出租方式（文本）
    'neighbourhood_cleansed',    # 所在街区（文本）
    'amenities'                  # 设施清单（文本，一整个字符串）
]
# 布尔列：取值只有't'/'f'，必须转成0/1才能喂给线性模型
bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'has_availability']

# 36个数值列：房东信息 + 房源规模 + 评分 + 位置 + 可订天数等
numeric_features = [
    'hosts_time_as_user_months',   # 房东注册为Airbnb用户的月数
    'hosts_time_as_host_months',   # 房东首次出租至今的月数
    'host_is_superhost',           # 是否超级房东（0/1）
    'host_has_profile_pic',        # 是否有头像（0/1）
    'host_identity_verified',      # 是否身份验证（0/1）
    'accommodates',                # 可容纳人数
    'bathrooms',                   # 浴室数量
    'bedrooms',                    # 卧室数量
    'beds',                        # 床的数量
    'minimum_nights',              # 最少入住天数
    'maximum_nights',              # 最多入住天数
    'host_listings_count',         # 房东挂的房源总数
    'review_scores_rating',        # 综合评分
    'latitude',                    # 纬度（位置）
    'longitude',                   # 经度（位置）
    'reviews_per_month',           # 每月评价数
    'number_of_reviews',           # 评价总数
    'number_of_reviews_ltm',       # 近12个月评价数
    'review_scores_accuracy',      # 描述准确性评分
    'review_scores_cleanliness',   # 清洁度评分
    'review_scores_checkin',       # 入住体验评分
    'review_scores_communication', # 沟通评分
    'review_scores_location',      # 位置评分
    'review_scores_value',         # 性价比评分
    'maximum_maximum_nights',      # 可预订最大上限
    'minimum_minimum_nights',      # 可预订最小下限
    'maximum_minimum_nights',      # 最短可订的最小值
    'minimum_maximum_nights',      # 最长可订的最小值
    'maximum_nights_avg_ntm',      # 平均最大可订天数
    'availability_30',             # 未来30天可订天数
    'availability_60',             # 未来60天可订天数
    'availability_90',             # 未来90天可订天数
    'availability_365',            # 未来365天可订天数
    'availability_eoy',            # 年底前可订天数
    'number_of_reviews_ly',        # 去年评价数
    'estimated_occupancy_l365d'    # 近365天估算入住率
]
print(f'X变量总数: {len(numeric_features) + len(categorical_features)}')
print(f'  说明：{len(numeric_features)}个数值列 + {len(categorical_features)}个文本列'
      f'（文本列独热编码后会展开成很多个0/1列）')

# 布尔列 t/f -> 0/1：.map({'t': 1, 'f': 0}) 把't'换成1、'f'换成0
for col in bool_cols:
    df_filtered[col] = df_filtered[col].map({'t': 1, 'f': 0})

# 重新构建 df_filtered1 和 X，让它们用上刚转好的0/1列
# y 和 groups 也必须同步重建：它们要和 X 来自完全相同的行，否则样本会错位
df_filtered1 = df_filtered.dropna(subset=numeric_features)
X = df_filtered1[numeric_features + categorical_features]
y = df_filtered1['price_clean']
groups = df_filtered1['host_id']


# ==================== 第8步：价格分布直方图 ====================
# 1行2列的布局：左图看原始价格分布，右图看对数变换后的分布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# 左图：原始价格直方图，bins=50切成50个等宽柱子
ax1.hist(df_filtered['price_clean'], bins=50, color='skyblue', edgecolor='black')
ax1.set_title('雅典价格分布（原值）')
ax1.set_xlabel('价格 (€)')
ax1.set_ylabel('房源数量')
ax1.grid(axis='y', alpha=0.3)
# 右图：np.log1p(价格) = ln(价格+1)，取对数让右偏严重的分布更对称
ax2.hist(np.log1p(df_filtered['price_clean']), bins=50, color='salmon', edgecolor='black')
ax2.set_title('对数变换后的价格分布')
ax2.set_xlabel('log(价格+1)')
ax2.set_ylabel('房源数量')
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/price_distribution.png', dpi=150)
print('>> 第1/5张图：价格分布直方图已保存。关闭图表窗口后继续...')
plt.show()


# ==================== 第9步：ColumnTransformer + 线性回归 分组交叉验证 ====================
# ColumnTransformer：数值列原样通过('passthrough')，分类文本列做独热编码(OneHotEncoder)
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Pipeline把"预处理 + 线性回归"串成一个整体，fit/predict一步到位
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

print('开始第9步：OneHotEncoder + 线性回归分组交叉验证（5折，大约需要3-5分钟）...')
# 记录每一折的RMSE（模型 vs 基线）
fold_rmses = []
baseline_rmses = []

# 分组K折交叉验证：每折训练模型、预测、和"猜均值"的基线对比
for fold, (train_idx, test_idx) in enumerate(gk5.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # --- 模型 ---
    model.fit(X_train, y_train)          # 在本折训练集上fit
    preds = model.predict(X_test)        # 在本折测试集上预测
    rmse = root_mean_squared_error(y_test, preds)   # 算测试RMSE
    fold_rmses.append(rmse)

    # --- "什么都不做"的基线：永远预测本折训练集的价格均值 ---
    train_mean = y_train.mean()          # 只用训练集的均值（和模型一样不能偷看测试集）
    baseline_preds = np.full_like(y_test, train_mean, dtype=float)  # 每个测试行都预测这个均值
    baseline_rmse = root_mean_squared_error(y_test, baseline_preds)
    baseline_rmses.append(baseline_rmse)

    print(
        f"第{fold + 1}折: 模型RMSE = €{rmse:.2f} | 基线RMSE = €{baseline_rmse:.2f} | 提升 = €{baseline_rmse - rmse:.2f}")

print("说明：上面每折的RMSE是'平均预测误差'（欧元），越小越好；"
      "'提升'=基线RMSE-模型RMSE，大于0说明模型比'瞎猜均价'强")
print(f"\n平均模型RMSE:    €{np.mean(fold_rmses):.2f}")
print("  说明：把文本列独热编码+数值列一起喂进线性回归，5折的平均误差")
print(f"平均基线RMSE: €{np.mean(baseline_rmses):.2f}")
print("  说明：'永远猜训练集均价'这个最笨策略的误差，是模型的及格线")


# ==================== 第10步：标准化数值特征 + 相关性热力图 ====================
print('开始第10步：标准化36个数值列、画热力图...')
# 下面开始分析"哪些变量最有价值"，只用数值列（36个，含已转0/1的布尔列）
# 文本/分类列不进惩罚回归和热力图
X_num = df_filtered1[numeric_features]
y_num = df_filtered1['price_clean']
groups_num = df_filtered1['host_id']

# 先把分组K折的折切好，存成(train_idx, test_idx)对列表
# 之所以不直接传groups=给RidgeCV/LassoCV：本机sklearn版本需要额外配置才支持，
# 用"折列表"当cv=既简单可靠，又保持了"按房东分组不拆散"的纪律
folds = [(tr, te) for tr, te in gk5.split(X_num, y_num, groups_num)]

# StandardScaler：把每列变成均值=0、标准差=1，消除量纲差异，让惩罚项公平对待每列
scaler = StandardScaler().fit(X_num)
Z = scaler.transform(X_num)

# np.corrcoef(Z.T) 算36列两两之间的相关系数，得到36×36相关矩阵
# 热力图：红=正相关、蓝=负相关、白=无关
C = np.corrcoef(Z.T)
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(C, vmin=-1, vmax=1, cmap='RdBu_r')
ax.set_xticks(range(len(numeric_features)))
ax.set_xticklabels(numeric_features, rotation=90, fontsize=7)
ax.set_yticks(range(len(numeric_features)))
ax.set_yticklabels(numeric_features, fontsize=7)
ax.set_title('标准化后36个数值特征的相关性')
fig.colorbar(im, shrink=0.8)
plt.tight_layout()
plt.savefig('figures/heatmap.png', dpi=150)
print('>> 第2/5张图：相关性热力图已保存。关闭图表窗口后继续...')
plt.show()


# ==================== 第11步：Ridge 回归，找出最有价值的变量 ====================
print('开始第11步：Ridge回归 + 交叉验证选lambda（100个lambda x 5折 = 500次拟合）...')
# 100个对数等距的lambda，从10^-3到10^4（D3课堂模板的网格）
grid = np.logspace(-3, 4, 100)
# RidgeCV 在分组交叉验证(folds)下自动挑lambda，scoring写明用RMSE（默认是R²，容易误导）
ridge_c = RidgeCV(alphas=grid, cv=folds,
                  scoring='neg_root_mean_squared_error').fit(Z, y_num)

# CV曲线：每个lambda下的5折交叉验证RMSE，直观看出lambda选在哪最合适
curve = np.array([-cross_validate(Ridge(alpha=a), Z, y_num, cv=folds,
                                  scoring='neg_root_mean_squared_error'
                                  )['test_score'].mean() for a in grid])
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(grid, curve)
ax.axvline(ridge_c.alpha_, ls='--', lw=1)   # 虚线标出RidgeCV选中的lambda
ax.set_xscale('log'); ax.set_xlabel('lambda (对数刻度)')
ax.set_ylabel('5折交叉验证RMSE (€)')
plt.tight_layout()
plt.savefig('figures/ridge_cv_curve.png', dpi=150)
print('>> 第3/5张图：Ridge CV曲线已保存。关闭图表窗口后继续...')
plt.show()

# 输出选中的lambda、CV误差、以及最大的|系数|
pos = int(np.where(grid == ridge_c.alpha_)[0][0]) + 1
print(f'[ridge] 选中的lambda {ridge_c.alpha_:.3f}  (网格第{pos}/100位)')
print('  说明：lambda是正则强度，RidgeCV自动选出的最优值；对应CV曲线图上虚线的位置')
print(f'[ridge] 交叉验证RMSE {curve.min():.3f} €  最大|系数| {np.abs(ridge_c.coef_).max():.2f}')
print('  说明：CV RMSE是「标准化后的36个数值列」做岭回归的平均误差，'
      '和上一节的值可直接对比；最大|系数|是影响力最大的那个变量，越大越重要')

# Ridge不把系数压成0，所以用|系数|大小衡量变量重要性，取最大的5个
ridge_top5 = [numeric_features[i] for i in np.argsort(-np.abs(ridge_c.coef_))[:5]]
print('[ridge] 五个|系数|最大的变量:')
print('  （系数都是标准化尺度：正号=价格随该变量上升，负号=价格随该变量下降）')
for name in ridge_top5:
    j = numeric_features.index(name)
    print(f'  {name:32s} {ridge_c.coef_[j]:+8.3f}')

# Ridge系数条形图：条越长说明该变量越有价值
# 只画|系数|最大的15个（36个全画太挤，下半截系数都接近0看不出名堂）
n_show = 15
order = np.argsort(np.abs(ridge_c.coef_))[-n_show:]
fig, ax = plt.subplots(figsize=(8, 6))
# top5用橙色突出，其余用蓝色
colors = ['C1' if numeric_features[i] in ridge_top5 else 'C0' for i in order]
ax.barh(np.array(numeric_features)[order], np.abs(ridge_c.coef_)[order], color=colors)
ax.set_xlabel('|系数| (标准化后)')
ax.set_title(f'Ridge: |系数|最大的{n_show}个变量（橙色=top5）')
plt.tight_layout()
plt.savefig('figures/ridge_coefs.png', dpi=150)
print('>> 第4/5张图：Ridge系数条形图已保存。关闭图表窗口后继续...')
plt.show()


# ==================== 第12步：Lasso 回归，稀疏地挑出最有价值的变量 ====================
print('开始第12步：Lasso回归（坐标下降法，最大50000次迭代）...')
# LassoCV会自动生成自己的100个lambda网格，交叉验证后refit
# max_iter=50_000给坐标下降法足够的迭代空间（D3课堂模板）
lasso_c = LassoCV(alphas=100, eps=1e-3, cv=folds,
                  max_iter=50_000, random_state=0).fit(Z, y_num)
print(f'\n[lasso] 选中的lambda {lasso_c.alpha_:.4f}')
print('  说明：LassoCV自己搜出来的最优lambda，越小模型越倾向用更多变量')
# 幸存者：系数没被压成0的列数，这就是"最有价值"的候选集合
print(f'[lasso] 幸存变量: {(lasso_c.coef_ != 0).sum()} / {len(numeric_features)}')
print('  说明：Lasso会把不重要的变量系数直接压成0，'
      '幸存下来的才是「最值钱」的那批变量')

# Lasso短名单：|系数|最大的5个列（D3课堂Part 2的写法）
top5 = [numeric_features[i] for i in np.argsort(-np.abs(lasso_c.coef_))[:5]]
print('[lasso] 五个|系数|最大的变量:')
print('  （正号=涨价因素，负号=压价因素；和Ridge的top5对照，'
      '两份名单越一致结论越可信）')
for name in top5:
    j = numeric_features.index(name)
    print(f'  {name:32s} {lasso_c.coef_[j]:+8.3f}')

# 系数路径图：每个变量的系数随lambda变化，top5用粗线标出
# 读法：lambda越小（越靠右）变量逐个"进入"模型
path = np.array([Lasso(alpha=a, max_iter=50_000).fit(Z, y_num).coef_
                 for a in lasso_c.alphas_])
fig, ax = plt.subplots(figsize=(10, 5))
for j in range(len(numeric_features)):
    hot = numeric_features[j] in top5
    ax.plot(lasso_c.alphas_, path[:, j],
            lw=1.6 if hot else 0.5, alpha=1.0 if hot else 0.35,
            label=numeric_features[j] if hot else None)
ax.axvline(lasso_c.alpha_, ls='--', lw=1)
ax.set_xscale('log'); ax.set_xlabel('lambda (对数刻度)')
ax.set_ylabel('系数（标准化尺度）')
ax.set_title('Lasso系数路径：每条线一个变量，粗线+图例=top5')
ax.legend(fontsize=8, loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig('figures/lasso_paths.png', dpi=150)
print('>> 第5/5张图：Lasso系数路径已保存。关闭图表窗口后结束。')
plt.show()


# ==================== 第13步：稳定性检验 —— 短名单换个数据还成立吗 ====================
# 随机抽80%的行重新fit lasso，看top5短名单有多少存活
# 逻辑：如果短名单是真实信号，丢掉20%的数据不应该改变它
# 缩放器在子样本上重新fit，分组折也在子样本上重新切（D3强调的纪律）
sub = np.random.default_rng(1).choice(len(Z), size=int(0.8 * len(Z)), replace=False)
sc_b = StandardScaler().fit(X_num.iloc[sub])
Zb = sc_b.transform(X_num.iloc[sub])
yb = y_num.iloc[sub]
folds_b = [(tr, te) for tr, te in gk5.split(Zb, yb, groups_num.iloc[sub])]
lasso_b = LassoCV(alphas=100, eps=1e-3, cv=folds_b,
                  max_iter=50_000, random_state=0).fit(Zb, yb)
top5_b = [numeric_features[i] for i in np.argsort(-np.abs(lasso_b.coef_))[:5]]
print(f'\n[稳定性] 80%重采样: 幸存变量 {(lasso_b.coef_ != 0).sum()}/{len(numeric_features)}; '
      f'top5中有 {len(set(top5) & set(top5_b))} 个仍在top5')
print('  说明：随机抽80%的数据重跑一遍Lasso。top5里若有4-5个存活，'
      '说明这份名单是稳定信号，不是靠运气选出来的')
gone = sorted(set(top5) - set(top5_b))
new = sorted(set(top5_b) - set(top5))
print('从短名单掉出的:', ', '.join(gone) or '无')
print('新进入短名单的:', ', '.join(new) or '无')
print('  说明：被挤掉/新进来的变量名；如果两条都是"无"，说明这份短名单非常稳定')
print('\n===== 全部完成！最终图表已保存到 figures/ 文件夹 =====')

