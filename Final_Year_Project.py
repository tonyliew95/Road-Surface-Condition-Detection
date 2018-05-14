import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import Program as pg

 
class App(QWidget):
 
    #widget
    def __init__(self):
        super().__init__()
        self.title = 'Road Surface Condition Detection'
        self.left = 640
        self.top = 480
        self.width = 640
        self.height = 480
        self.initUI()
 
    #UI initiation
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.button_globalThreshold()
        self.button_Otsu()
        self.button_Canny()
        self.button_K_means()
        self.button_haar()
        self.show()

    #select folder 
    def openFileNameDialog1(self):    
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder")
        p = pg.Global_Threshold(fileName)

    def openFileNameDialog2(self):    
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder")
        p = pg.Otsu_Detection(fileName)

    def openFileNameDialog3(self):    
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder")
        p = pg.Canny_detection(fileName)

    def openFileNameDialog4(self):    
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder")
        p = pg.Kmeans_detection(fileName)
    #select folder & xml file
    def openFileNameDialog5(self):    
        fileName = QFileDialog.getExistingDirectory(self,"Select Folder")
        data_file,_ = QFileDialog.getOpenFileName(self,"Select XML","","*xml")
        print(data_file)
        p = pg.haarClasifier_image(fileName,data_file)
 

    def button_globalThreshold(self):
        button = QPushButton('Global Thresholding', self)#create button
        button.move(280,200) #button position
        button.clicked.connect(self.openFileNameDialog1)#connect to function openFileNameDialog1


    def button_Otsu(self):
        button = QPushButton('Otsu Thresholding', self)#create button
        button.move(280,240) #button position
        button.clicked.connect(self.openFileNameDialog2)#connect to function openFileNameDialog2

    def button_Canny(self):
        button = QPushButton('Canny', self)#create button
        button.move(280,280) #button position
        button.clicked.connect(self.openFileNameDialog3)#connect to function openFileNameDialog3

    def button_K_means(self):
        button = QPushButton('K_mean', self)#create button
        button.move(280,320) #button position
        button.clicked.connect(self.openFileNameDialog4)#connect to function openFileNameDialog4

    def button_haar(self):
        button = QPushButton('Haar-cascade classifier', self)#create button
        button.move(280,360) #button position
        button.clicked.connect(self.openFileNameDialog5)#connect to function openFileNameDialog5


    @pyqtSlot()
    def on_click(self):
        print('PyQt5 button click')

 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())