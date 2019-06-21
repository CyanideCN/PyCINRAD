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
        self.main_window = MainWindow
        self.graphicscene = QtWidgets.QGraphicsScene()
        self.button_table = {'REF':self.radioButton, 'VEL':self.radioButton_2, 'RHO':self.radioButton_3,
                             'ZDR':self.radioButton_4, 'SW':self.radioButton_5, 'PHI':self.radioButton_6,
                             'ET':self.radioButton_7, 'VIL':self.radioButton_8, 'CR':self.radioButton_9}
        for k in self.button_table.keys():
            self.button_table[k].clicked.connect(partial(self.on_button_activate, k))
        self.actionOpen.triggered.connect(self._open)
        self.actionClose.triggered.connect(self._close)
        self.pushButton_2.clicked.connect(self.draw)
        self.comboBox.activated.connect(self.on_combobox_activate)
        self.comboBox.currentIndexChanged.connect(self.on_combobox_changed)
        self.pushButton.clicked.connect(self.on_textbox_update)
        for i in self.button_table.values():
            i.setEnabled(False)

    def _open(self):
        f = QtWidgets.QFileDialog.getOpenFileName()
        fn = f[0]
        if fn == '':
            return
        if fn == self.last_fname:
            if hasattr(self, 'cinrad'):
                pass
        else:
            try:
                self.cinrad = read(fn)
            except Exception:
                self._message('无法读取该数据')
                return
            self.last_fname = fn
        # Display basic info
        self._flush()
        info = '站名:{}\n扫描时间:{}'
        self.basic_info_string = info.format(self.cinrad.name, self.cinrad.scantime.strftime('%Y-%m-%d %H:%M:%SZ'))
        self.label_3.setText(self.basic_info_string)
        # Extract available tilts and display in menu
        self.comboBox.addItems(['仰角{}-{:.2f}°'.format(i[0], i[1]) for i in enumerate(self.cinrad.el)])

    def _flush(self):
        self.comboBox.clear()
        for i in self.button_table.values():
            i.setEnabled(False)
        self.label_3.setText('')
        self.graphicscene.clear()
    
    def on_combobox_activate(self, index):
        self.tilt = index

    def on_combobox_changed(self):
        for i in self.button_table.values():
            i.setEnabled(False)
        if hasattr(self, 'cinrad'):
            ap = self.cinrad.available_product(self.tilt)
            av = self.button_table.keys()
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
        self.graphicscene.addWidget(dr)
        self.graphicsView.setScene(self.graphicscene)
        self.graphicsView.show()
        self.redraw = True

    def _close(self):
        plt.close('all')
        self.redraw = False
        del self.cinrad
        self._flush()

    def _message(self, message):
        msg = QtWidgets.QMessageBox.warning(self.main_window, 'Error!', message)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = RadarUI()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())