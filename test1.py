from PySide2.QtWidgets import QApplication, QMessageBox, QPushButton, QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile

class Recognition:
    def __init__(self):
        self.FlagVideo = False
        self.FlagEeg = False
        self.PathEeg = ''
        self.PathVideo = ''
        self.init()


    def init(self):
        # 从文件中加载UI
        qfile_stats = QFile("ori_ui/main_ui.ui")
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()

        # 从 UI 定义中动态创建一个相应的窗口对象
        self.ui = QUiLoader().load(qfile_stats)

        # 正确的做法是使用lambda或functools.partial，以确保连接信号时只传递一个可调用对象，而不是执行函数。
        self.ui.BtnSelectEeg.clicked.connect(lambda: self.showDialog('SelectEegFile'))
        self.ui.BtnSelectVideo.clicked.connect(lambda: self.showDialog('SelectVideoFile'))
        self.ui.BtnVideoRec.clicked.connect(lambda: self.VideoRec)
        self.ui.BtnEegRec.clicked.connect(lambda: self.EegRec)
        self.ui.BtnMultiRec.clicked.connect(lambda: self.MultiRec)

        # 按钮初始化不可见
        self.ui.BtnVideoRec.setEnabled(False)
        self.ui.BtnEegRec.setEnabled(False)
        self.ui.BtnMultiRec.setEnabled(False)

        self.ui.show()

    # input: 用户选择文件路径 output: 用户选择的文件路径
    # 覃智科
    def showDialog(self, FileStyle):
        fnames, _ = QFileDialog.getOpenFileNames(self.ui,'open file','D:\\Postgraduate\\SHU-AI\\Professor_Wang\\Projects\\EEG_Multimodal_Project\\code\\EEG_Pic_EmoRec\\extract_data')
        if fnames[0]:
            fname = fnames[0]
            if FileStyle == 'SelectEegFile':
                self.ui.FileNameEeg.setText(fname)
                self.FlagEeg = True
                self.PathEeg = fname
            elif FileStyle == 'SelectVideoFile':
                self.ui.FileNameVideo.setText(fname)
                self.FlagVideo = True
                self.PathVideo = fname
        if self.FlagVideo == True:
            self.ui.BtnVideoRec.setEnabled(True)
        if self.FlagEeg == True:
            self.ui.BtnEegRec.setEnabled(True)
        if self.FlagEeg == True and self.FlagVideo == True:
            self.ui.BtnMultiRec.setEnabled(True)

    # input: self.PathEeg 和 Self.PathVideo
    # output: self.VideoArray self.EegRec ---给ResAnalysis用
    # 王肖
    def VideoRec(self):
        pass
    # 王肖
    def EegRec(self):
        pass
    # 王肖
    def MultiRec(self):
        pass
    # 覃智科
    def ResAnalysis(self):
        pass
app = QApplication([])
Rec = Recognition()
Rec.ui.show()
app.exec_()