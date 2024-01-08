from pyecharts.charts import Bar
import pyecharts.options as opts
from pyecharts.charts import Line
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5 import uic
import sys
from PyQt5.QtWebEngineWidgets import QWebEngineView
from pyecharts.commons.utils import JsCode
from model.SVM_emotion_recognition import SVM
from model.CV.predicted import cnn_predicted
from EEG_processing.processing import EEG_processing
# 参考网站：https://gallery.pyecharts.org/#/Line/temperature_change_line_chart



class Recognition:
    def __init__(self):
        self.FlagPicture = False
        self.FlagEeg_single = False
        self.FlagEeg_continue = False
        self.PathEeg_single = '' # 脑电文件路径
        self.PathEeg_continue = ''  # 脑电文件路径
        self.PathPicture = ''
        self.init()

    def init(self):
        # 从文件中加载UI
        self.ui = uic.loadUi('D:/table/EEG_Pic_EmoRec/ori_ui/main_ui.ui')

        # 正确的做法是使用lambda或functools.partial，以确保连接信号时只传递一个可调用对象，而不是执行函数。
        self.ui.BtnSelectEeg_single.clicked.connect(lambda: self.showDialog('BtnSelectEeg_single'))
        self.ui.BtnSelectEeg_continue.clicked.connect(lambda: self.showDialog('BtnSelectEeg_continue'))
        self.ui.BtnSelectPicture.clicked.connect(lambda: self.showDialog('BtnSelectPicture'))

        self.ui.BtnPictureRec.clicked.connect(self.PictureRec)
        self.ui.BtnEegRec_single.clicked.connect(self.EegRec_single)
        self.ui.BtnEegRec_continue.clicked.connect(self.EegRec_continue)
        self.ui.BtnMultiRec.clicked.connect(self.MultiRec)

        # 按钮初始化不可见
        self.ui.BtnPictureRec.setEnabled(False)
        self.ui.BtnEegRec_continue.setEnabled(False)
        self.ui.BtnEegRec_single.setEnabled(False)
        self.ui.BtnMultiRec.setEnabled(False)

        self.ui.show()

    # input: 用户选择文件路径 output: 用户选择的文件路径
    # 覃智科
    def showDialog(self, FileStyle):
        fnames, _ = QFileDialog.getOpenFileNames(self.ui, 'open file',
                                                 r'D:\table\EEG_Pic_EmoRec\Data\软件测试数据')
        if fnames:
            if fnames[0]:
                fname = fnames[0]
                if FileStyle == 'BtnSelectEeg_single':
                    self.ui.FileNameEeg_single.setText(fname)
                    self.FlagEeg_single = True
                    self.PathEeg_single = fname
                elif FileStyle == 'BtnSelectEeg_continue':
                    self.ui.FileNameEeg_continue.setText(fname)
                    self.FlagEeg_continue = True
                    self.PathEeg_continue = fname
                elif FileStyle == 'BtnSelectPicture':
                    self.ui.FileNamePicture.setText(fname)
                    self.FlagPicture = True
                    self.PathPicture = fname
            if self.FlagPicture == True:
                self.ui.BtnPictureRec.setEnabled(True)
            if self.FlagEeg_single == True:
                self.ui.BtnEegRec_single.setEnabled(True)
            if self.FlagEeg_continue == True:
                self.ui.BtnEegRec_continue.setEnabled(True)
            if (self.FlagEeg_continue == True and self.FlagPicture == True) or (self.FlagEeg_single == True and self.FlagPicture == True):
                self.ui.BtnMultiRec.setEnabled(True)

    # input: self.PathEeg 和 Self.PathVideo
    # output: self.VideoArray self.EegRec ---给ResAnalysis用
    # 王肖
    def PictureRec(self):
        # self.load_echarts()
        # pass
        result_dict=cnn_predicted(self.PathPicture)
        print(result_dict)
        EEG_processing('Anger_EGG_1.fif',"") ##生成两张图片，用于前端展示，分别是raw_eeg_plot.png，和topography_plot.png
        """
        {'Sad': 0.028197653591632843, 'Digust': 1.393415089978589e-07, 'Happy': 3.258527794969268e-05, 
        'Fear': 0.03532076254487038, 'Surprise': 0.0024856592062860727, 'Neutral': 0.9339504241943359, 'Anger': 1.2757378499372862e-05}
        可视化展示
        """




    # 王肖
    def EegRec_single(self):
        svm = SVM()

        result_dict = svm.svm_singal_analysics(self.PathEeg_single)

        """
        {'Anger': '2.13%', 'Disgust': '5.78%', 'Fear': '0.29%', 'Happy': '0.78%', 'Neutral': '22.20%', 'Sad': '68.72%', 'Surprise': '0.10%'}
        """
        print(result_dict)


        # 画个柱状图
        color_function = """
                function (params) {
                    if (params.name == 'Anger') 
                        return 'red';
                    else if (params.name=='Disgust') 
                        return 'black';
                    else if (params.name=='Fear') 
                    return 'purple';
                    else if (params.name=='Happy') 
                    return 'orange';
                    else if (params.name=='Neutral') 
                    return 'pink';
                    else if (params.name=='Sad') 
                    return 'brown';
                    else return 'green';
                }
                """
        # 创建柱状图
        chart = (
            Bar()
                .add_xaxis(['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
                .add_yaxis('', [result_dict['Anger'], result_dict['Disgust'], result_dict['Fear'],
                                       result_dict['Happy'], result_dict['Neutral'], result_dict['Sad'],
                                       result_dict['Surprise']],category_gap="60%",itemstyle_opts=opts.ItemStyleOpts(color=JsCode(color_function)))
                .set_global_opts(title_opts=opts.TitleOpts(title="单点脑电识别", subtitle="Timi"))
                .set_series_opts(
                label_opts=opts.LabelOpts(
                    is_show=True,
                    position="top",
                    formatter=JsCode("function (params) {return params.value + '%'}")
                )
            )
        )
        chart.render(r'D:\table\EEG_Pic_EmoRec\EegRec_single_bar.html')
        with open(r"D:\table\EEG_Pic_EmoRec\EegRec_single_bar.html", "r", encoding="utf-8") as file:
            html_content = file.read()
            print("get html file!!")
            # 将 HTML 页面加载到 QWebEngineView 中
        self.ui.webEngineView.setHtml(html_content)

    def EegRec_continue(self):
        self.render_Eeg_continue_Emotion_Chart()


    # 王肖
    def MultiRec(self):
        svm = SVM()
        result_list_eeg = svm.svm_singal_analysics(self.PathEeg_single)
        result_dict_pc=cnn_predicted(self.PathPicture)

        # 加权融合
        result_dict = {}

        # SVM预测的权重
        weight_svm = 0.7
        # CNN预测的权重
        weight_cnn = 0.3

        # 遍历每个情绪类别
        for emotion in result_list_eeg.keys():
            # 使用指定的权重结合预测概率
            combined_prob = (weight_svm * result_list_eeg[emotion]) + (weight_cnn * result_dict_pc[emotion])*100

            # 存储合并的结果
            result_dict[emotion] = round(combined_prob, 2)
        # 打印最终的加权融合结果
        print(result_dict)
        # 画个柱状图
        color_function = """
                      function (params) {
                          if (params.name == 'Anger') 
                              return 'red';
                          else if (params.name=='Disgust') 
                              return 'black';
                          else if (params.name=='Fear') 
                          return 'purple';
                          else if (params.name=='Happy') 
                          return 'orange';
                          else if (params.name=='Neutral') 
                          return 'pink';
                          else if (params.name=='Sad') 
                          return 'brown';
                          else return 'green';
                      }
                      """
        # 创建柱状图
        chart = (
            Bar()
            .add_xaxis(['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
            .add_yaxis('', [result_dict['Anger'], result_dict['Disgust'], result_dict['Fear'],
                                   result_dict['Happy'], result_dict['Neutral'], result_dict['Sad'],
                                   result_dict['Surprise']], category_gap="60%",
                       itemstyle_opts=opts.ItemStyleOpts(color=JsCode(color_function)))
            .set_global_opts(title_opts=opts.TitleOpts(title="单点脑电识别", subtitle="Timi"))
            .set_series_opts(
                label_opts=opts.LabelOpts(
                    is_show=True,
                    position="top",
                    formatter=JsCode("function (params) {return params.value + '%'}")
                )
            )
        )
        chart.render(r'D:\table\EEG_Pic_EmoRec\MultiRec_bar.html')
        with open(r"D:\table\EEG_Pic_EmoRec\MultiRec_bar.html", "r", encoding="utf-8") as file:
            html_content = file.read()
            print("get html file!!")
            # 将 HTML 页面加载到 QWebEngineView 中
        self.ui.webEngineView.setHtml(html_content)
        """
        {'Anger': 2.34, 'Disgust': 4.05, 'Fear': 0.2, 'Happy': 1.61, 'Neutral': 15.61, 'Sad': 76.12, 'Surprise': 0.07}
        """

    # 覃智科
    def render_Eeg_continue_Emotion_Chart(self):
        svm=SVM()
        print(self.PathEeg_continue)
        result_list=svm.svm_more_analysics(self.PathEeg_continue)
        # for i in result_list:
        #     print(i)
        print('模型初始化完成……')
        Surprise = []
        Sad = []
        Happy = []
        Neutral = []
        Anger = []
        Fear = []
        Disgust = []
        print('数据加载中……')
        for result in result_list:
            Surprise.append(result['Surprise'])
            Sad.append(result["Sad"])
            Happy.append(result["Happy"])
            Neutral.append(result["Neutral"])
            Anger.append(result["Anger"])
            Fear.append(result["Fear"])
            Disgust.append(result["Disgust"])
        length=len(result_list)
        time_list = list(range(0, length, 1))

        print('渲染前……')
        chart = (
            Line()
                .add_xaxis(xaxis_data=time_list)
                .add_yaxis(
                series_name="惊讶",
                y_axis=Surprise,
                label_opts=opts.LabelOpts(is_show=False),
                is_smooth=True,  # 添加平滑曲线的参数
                linestyle_opts=opts.LineStyleOpts(width=3),
            )
                .add_yaxis(
                series_name="难过",
                y_axis=Sad,
                label_opts=opts.LabelOpts(is_show=False),
                is_smooth=True,  # 添加平滑曲线的参数
                linestyle_opts=opts.LineStyleOpts(width=3),

            )
                .add_yaxis(
                series_name="厌恶",
                y_axis=Disgust,
                label_opts=opts.LabelOpts(is_show=False),
                is_smooth=True,  # 添加平滑曲线的参数
                linestyle_opts=opts.LineStyleOpts(width=3),

            )
                .add_yaxis(
                series_name="高兴",
                y_axis=Happy,
                label_opts=opts.LabelOpts(is_show=False),
                is_smooth=True,  # 添加平滑曲线的参数
                linestyle_opts=opts.LineStyleOpts(width=3),

            )
                .add_yaxis(
                series_name="害怕",
                y_axis=Fear,
                label_opts=opts.LabelOpts(is_show=False),
                is_smooth=True,  # 添加平滑曲线的参数
                linestyle_opts=opts.LineStyleOpts(width=3),

            )
                .add_yaxis(
                series_name="愤怒",
                y_axis=Anger,
                label_opts=opts.LabelOpts(is_show=False),
                is_smooth=True,  # 添加平滑曲线的参数
                linestyle_opts=opts.LineStyleOpts(width=3),

            )
                .add_yaxis(
                series_name="中性",
                y_axis=Neutral,
                label_opts=opts.LabelOpts(is_show=False),
                is_smooth=True,  # 添加平滑曲线的参数
                linestyle_opts=opts.LineStyleOpts(width=3),

            )
                .set_global_opts(
                title_opts=opts.TitleOpts(title="微重力下的脑电情绪分析结果", subtitle="纯属虚构"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                toolbox_opts=opts.ToolboxOpts(is_show=True),
                xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False,name="10s/次"),
                yaxis_opts=opts.AxisOpts(type_="value",name="预测百分比",axislabel_opts=opts.LabelOpts(formatter="{value}%"),)
            )

        )
        # 直接获取渲染后的HTML字符串
        # html_content = chart.render_embed()
        # return html_content
        print('正在渲染……')
        chart.render(r"D:\table\EEG_Pic_EmoRec\temperature_change_line_chart.html")
        print('渲染成功！')
        with open(r"D:\table\EEG_Pic_EmoRec\temperature_change_line_chart.html", "r", encoding="utf-8") as file:
            html_content = file.read()
            print("get html file!!")
        # 将 HTML 页面加载到 QWebEngineView 中
        self.ui.webEngineView.setHtml(html_content)

    # def load_echarts(self):
    #     # 构建 HTML 页面，加载 echarts 图表
    #     self.render_Eeg_continue_Emotion_Chart()
    #     # webbrowser.open('./temperature_change_line_chart.html')
    #     with open("temperature_change_line_chart.html", "r", encoding="utf-8") as file:
    #         html_content = file.read()
    #         print("get html file!!")
    #     # 将 HTML 页面加载到 QWebEngineView 中
    #     self.ui.webEngineView.setHtml(html_content)


        # # 构建 HTML 页面，加载 echarts 图表
        # html_content = self.render_emotions_chart()

        # 将 HTML 字符串加载到 QWebEngineView 中
        self.ui.webEngineView.setHtml(html_content)


if __name__ == '__main__':
    app = QApplication([])
    Rec = Recognition()
    Rec.ui.show()
    sys.exit(app.exec_())
