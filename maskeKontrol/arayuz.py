
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(300, 169)
        MainWindow.setMinimumSize(QtCore.QSize(300, 169))
        MainWindow.setMaximumSize(QtCore.QSize(300, 169))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.kameraAc = QtWidgets.QPushButton(self.centralwidget)
        self.kameraAc.setGeometry(QtCore.QRect(20, 50, 251, 61))
        self.kameraAc.setObjectName("kameraAc")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 291, 51))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 300, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("serkan", "serkan"))
        self.kameraAc.setText(_translate("serkan", "KAMERA"))
        self.label.setText(_translate("serkan", "maske kontrolü için butona basınız"))
