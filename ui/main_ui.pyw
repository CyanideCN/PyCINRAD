import sys
from functools import partial

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from PyQt5 import QtWidgets, QtCore
import cinrad
from ui_struct import Ui_MainWindow

def read(fpath):
    try:
        f = cinrad.io.CinradReader(fpath)
    except Exception:
        f = cinrad.io.StandardData(fpath)
    return f

class Figure_Canvas(FigureCanvas):
    def __init__(self, parent=None, width=9, height=9, dpi=90):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

class RadarUI(Ui_MainWindow):
    '''Further modification from infrastructure'''
    def __init__(self):
        self.last_fname = None
        self.tilt = 0
        self.dtype = 'REF'
        self.drange = 230
        self.redraw = False

    def setupUi(self, MainWindow):
        super(RadarUI, self).setupUi(MainWindow)
        self.button_table = {'REF':self.radioButton, 'VEL':self.radioButton_2, 'RHO':self.radioButton_3,
                             'ZDR':self.radioButton_4, 'SW':self.radioButton_5, 'PHI':self.radioButton_6,
                             'ET':self.radioButton_7, 'VIL':self.radioButton_8, 'CR':self.radioButton_9}
        for k in self.button_table.keys():
            self.button_table[k].clicked.connect(partial(self.on_button_activate, k))
        self.actionOpen.triggered.connect(self._open)
        self.pushButton_2.clicked.connect(self.draw)
        self.comboBox.currentIndexChanged.connect(self.on_combobox_activate)
        self.pushButton.clicked.connect(self.on_textbox_update)

    def _open(self):
        f = QtWidgets.QFileDialog.getOpenFileName()
        fn = f[0]
        if fn == '':
            return
        if fn == self.last_fname:
            if hasattr(self, 'cinrad'):
                pass
        else:
            self.cinrad = read(fn)
            self.last_fname = fn
        # Display basic info
        info = '站名:{}\n扫描时间:{}'
        self.basic_info_string = info.format(self.cinrad.name, self.cinrad.scantime.strftime('%Y-%m-%d %H:%M:%SZ'))
        self.label_3.setText(self.basic_info_string)
        # Extract avaliable tilts and display in menu
        self.comboBox.clear()
        self.comboBox.addItems(['仰角{}-{:.2f}°'.format(i[0], i[1]) for i in enumerate(self.cinrad.el)])
    
    def on_combobox_activate(self, index):
        self.tilt = index
        ap = self.cinrad.avaliable_product(index)
        av = self.button_table.keys()
        for i in self.button_table.values():
            i.setEnabled(False)
        for p in ap:
            if p in av:
                self.button_table[p].setEnabled(True)

    def on_button_activate(self, prod):
        self.dtype = prod

    def on_textbox_update(self):
        rad = self.plainTextEdit.toPlainText()
        self.drange = float(rad)

    def draw(self):
        if self.redraw:
            plt.close('all')
        dr = Figure_Canvas()
        data = self.cinrad.get_data(self.tilt, self.drange, self.dtype)
        fig = cinrad.visualize.PPI(data, fig=dr.figure, plot_labels=False)
        graphicscene = QtWidgets.QGraphicsScene()
        graphicscene.addWidget(dr)
        self.graphicsView.setScene(graphicscene)
        self.graphicsView.show()
        self.redraw = True

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = RadarUI()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())