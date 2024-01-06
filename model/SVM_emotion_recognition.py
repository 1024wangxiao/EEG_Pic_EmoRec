import pandas as pd 
import joblib
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder

class SVM:
    def svm_singal_analysics(self,path):
        # 加载保存的最佳参数
        with open(r"D:\table\EEG_Pic_EmoRec\Data\model_params\SVM_best_params.txt") as f:
            best_params = eval(f.read())

        # 步骤1~步骤5,读取和预处理测试数据

        input_filename = path
        df = pd.read_excel(input_filename,engine='openpyxl',header=None)

        data=df.iloc[0,2:].values
        data=np.array([data])


        # 加载最佳模型
        loaded_model = joblib.load(r'D:\table\EEG_Pic_EmoRec\Data\model_params\best_svm_model.pkl')
        loaded_model.set_params(**best_params)
        # 使用加载的模型进行预测概率
        decision_values = loaded_model.decision_function(data)

        # 将决策值转换为概率（softmax函数）
        predicted_probabilities = softmax(decision_values, axis=1)

        # 获取类别标签
        class_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        # 将标签和概率一一对应
        #数据小数点表示
        # result_dict = {label: prob for label, prob in zip(class_labels, predicted_probabilities[0])}
        result_dict = {label: f'{prob * 100:.2f}%' for label, prob in zip(class_labels, predicted_probabilities[0])}
        result_dict = {key: float(value.rstrip('%')) for key, value in result_dict.items()}
        # 打印预测结果
        return result_dict


    def svm_more_analysics(self, path):
        # 加载保存的最佳参数
        with open(r"D:\table\EEG_Pic_EmoRec\Data\model_params\SVM_best_params.txt") as f:
            best_params = eval(f.read())

        # 步骤1~步骤5,读取和预处理测试数据

        input_filename = path
        df = pd.read_excel(input_filename, engine='openpyxl', header=None)

        data = df.iloc[1:, 2:].values
        data = np.array([data])
        data=np.squeeze(data, axis=0)
        # data=data.reshape((60, 252))
        results = []
        for i in data:
            i = i.reshape((1, 252))
            # 加载最佳模型
            loaded_model = joblib.load(r'D:\table\EEG_Pic_EmoRec\Data\model_params\best_svm_model.pkl')
            loaded_model.set_params(**best_params)
            # 使用加载的模型进行预测概率
            decision_values = loaded_model.decision_function(i)

            # 将决策值转换为概率（softmax函数）
            predicted_probabilities = softmax(decision_values, axis=1)
            # 获取类别标签
            class_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            # 将标签和概率一一对应
            # 数据小数点表示
            # result_dict = {label: prob for label, prob in zip(class_labels, predicted_probabilities[0])}
            result_dict = {label: f'{prob * 100:.2f}%' for label, prob in zip(class_labels, predicted_probabilities[0])}
            result_dict = {key: float(value.rstrip('%')) for key, value in result_dict.items()}
            results.append(result_dict)
        # 打印预测结果
        return results


if __name__=="__main__":
    pass
    #svm单个片段分析实例：返回各个情绪的识别准确率百分比
    # /mnt/data/252/SVM_best_params.txt
    svm=SVM()
    # result_dict=svm.svm_singal_analysics("/mnt/data/252/SVM_best_params.txt")
    # print(result_dict)
    result=svm.svm_more_analysics(r"D:\table\EEG_Pic_EmoRec\Data\text_continue.xlsx")
    for i in result:
        print(i)






