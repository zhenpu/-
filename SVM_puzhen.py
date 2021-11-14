import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split  # 用于数据集划分
from  sklearn.metrics import accuracy_score  # 用于评价预测准确率

df = pd.read_csv('E:\\实验数据\\400+500粒种子高光谱提取\\400\\400 PCA END.csv')  # 导入处理后的我拍摄的高光谱数据
X = df.drop(['VAR00001'], axis=1)  # x是变量，选出来的主成分，这一语句用于在数据集中去除标签
X_a = X.values
y = df['VAR00001']  # 定义标签
y_a = y.values

print(X_a)  # 我用来看看导入对了没
print(y_a)

train_data, test_data = train_test_split(X_a, random_state=10, train_size=0.8, test_size=0.2)  # 划分数据集
train_label, test_label = train_test_split(y_a, random_state=10, train_size=0.8, test_size=0.2)  # 划分数据标签
print(len(train_data))

classifier = svm.SVC(C=0.2, kernel='linear', gamma=10, decision_function_shape='ovr', random_state=1)  # 导入SVM并设置参数
classifier.fit(train_data, train_label.ravel())  # 训练模型

pre_train = classifier.predict(train_data)  # 预测训练集
pre_test = classifier.predict(test_data)  # 预测测试集

print(accuracy_score(train_label, pre_train))  # 输出训练集准确率
print(accuracy_score(test_label, pre_test))  # 输出测试集准确率
