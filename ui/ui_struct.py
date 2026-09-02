# -*- coding: utf-8 -*-
# PyCINRAD Radar Display UI - Updated for new cinrad interfaces
# Layout-based design with full feature support

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 850)
        MainWindow.setMinimumSize(1000, 700)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(2, 2, 2, 2)
        self.main_layout.setSpacing(2)

        # ===== Left Panel (Scroll Area + Bottom Bar) =====
        self.left_container = QtWidgets.QWidget(self.centralwidget)
        self.left_container.setObjectName("left_container")
        self.left_container_layout = QtWidgets.QVBoxLayout(self.left_container)
        self.left_container_layout.setContentsMargins(0, 0, 0, 0)
        self.left_container_layout.setSpacing(2)

        self.scrollArea = QtWidgets.QScrollArea(self.left_container)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setMinimumWidth(230)
        self.scrollArea.setMaximumWidth(280)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollContent = QtWidgets.QWidget()
        self.scrollContent.setObjectName("scrollContent")
        self.left_layout = QtWidgets.QVBoxLayout(self.scrollContent)
        self.left_layout.setContentsMargins(6, 6, 6, 6)
        self.left_layout.setSpacing(6)

        # --- Data Reader Selection ---
        self.reader_group = QtWidgets.QGroupBox("数据接口")
        self.reader_layout = QtWidgets.QVBoxLayout(self.reader_group)
        self.reader_combo = QtWidgets.QComboBox()
        self.reader_combo.setMinimumHeight(26)
        readers = [
            ("read_auto", "自动识别接口(默认)"),
            ("StandardData", "标准格式基数据"),
            ("StandardPUP", "标准PUP产品"),
            ("MocMosaic", "拼图V3.0(MocMosaic)"),
            ("SWAN", "SWAN3/风雷(SWAN)"),
            ("CinradReader", "老格式基数据(CinradReader)"),
            ("PhasedArrayData", "相控阵(PhasedArrayData)"),
        ]
        for key, tip in readers:
            self.reader_combo.addItem(key, key)
            self.reader_combo.setItemData(self.reader_combo.count() - 1, tip,
                                          QtCore.Qt.ToolTipRole)
        self.reader_combo.setCurrentIndex(0)  # default read_auto
        self.reader_args = QtWidgets.QLineEdit()
        self.reader_args.setPlaceholderText("参数, 如 radar_type=SA")
        self.reader_args.setClearButtonEnabled(True)
        self.reader_row = QtWidgets.QHBoxLayout()
        self.reader_row.addWidget(self.reader_combo, 3)
        self.reader_row.addWidget(self.reader_args, 2)
        self.reader_layout.addLayout(self.reader_row)
        self.left_layout.addWidget(self.reader_group)

        # --- Scan Info ---
        self.info_group = QtWidgets.QGroupBox("扫描信息")
        self.info_layout = QtWidgets.QVBoxLayout(self.info_group)
        self.label_3 = QtWidgets.QLabel("")
        self.label_3.setWordWrap(True)
        self.label_3.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.label_3.setMinimumHeight(60)
        self.info_layout.addWidget(self.label_3)
        self.left_layout.addWidget(self.info_group)

        # --- Tilt Selection ---
        self.tilt_group = QtWidgets.QGroupBox("仰角选择")
        self.tilt_layout = QtWidgets.QVBoxLayout(self.tilt_group)
        self.label_4 = QtWidgets.QLabel("选择仰角:")
        self.comboBox = QtWidgets.QComboBox()
        self.comboBox.setMinimumHeight(26)
        self.tilt_layout.addWidget(self.label_4)
        self.tilt_layout.addWidget(self.comboBox)
        self.left_layout.addWidget(self.tilt_group)

        # --- Height Selection (for 3D data) ---
        self.height_group = QtWidgets.QGroupBox("高度选择")
        self.height_layout = QtWidgets.QVBoxLayout(self.height_group)
        self.height_combo = QtWidgets.QComboBox()
        self.height_combo.setMinimumHeight(26)
        self.height_layout.addWidget(self.height_combo)
        self.left_layout.addWidget(self.height_group)

        # --- Product Selection ---
        self.prod_group = QtWidgets.QGroupBox("产品选择")
        self.prod_layout = QtWidgets.QVBoxLayout(self.prod_group)
        self.prod_combo = QtWidgets.QComboBox()
        self.prod_combo.setMinimumHeight(26)
        self.prod_layout.addWidget(self.prod_combo)
        # Calculated products
        self.calc_label = QtWidgets.QLabel("计算产品:")
        self.calc_combo = QtWidgets.QComboBox()
        self.calc_combo.addItems(["不计算", "CR组合反射率", "ET回波顶高", "VIL垂直积分液态水", "VILD密度VIL", "HCL水凝物分类"])
        self.prod_layout.addWidget(self.calc_label)
        self.prod_layout.addWidget(self.calc_combo)
        self.left_layout.addWidget(self.prod_group)

        # --- Plot Parameters ---
        self.param_group = QtWidgets.QGroupBox("绘图参数")
        self.param_layout = QtWidgets.QFormLayout(self.param_group)
        self.label = QtWidgets.QLabel("绘图半径(km):")
        self.plainTextEdit = QtWidgets.QLineEdit("230")
        self.plainTextEdit.setMaximumWidth(120)
        self.pushButton = QtWidgets.QPushButton("更新")
        self.param_layout.addRow(self.label, self.plainTextEdit)
        self.dpi_label = QtWidgets.QLabel("DPI:")
        self.dpi_input = QtWidgets.QLineEdit("300")
        self.dpi_input.setMaximumWidth(120)
        self.param_layout.addRow(self.dpi_label, self.dpi_input)
        self.left_layout.addWidget(self.param_group)

        # --- Extent ---
        self.extent_group = QtWidgets.QGroupBox("经纬度范围 (可选)")
        self.extent_layout = QtWidgets.QFormLayout(self.extent_group)
        self.extent_check = QtWidgets.QCheckBox("自定义范围")
        self.lon_min = QtWidgets.QLineEdit()
        self.lon_max = QtWidgets.QLineEdit()
        self.lat_min = QtWidgets.QLineEdit()
        self.lat_max = QtWidgets.QLineEdit()
        for w in [self.lon_min, self.lon_max, self.lat_min, self.lat_max]:
            w.setMaximumWidth(120)
        self.extent_layout.addRow(self.extent_check)
        self.extent_layout.addRow("经度最小:", self.lon_min)
        self.extent_layout.addRow("经度最大:", self.lon_max)
        self.extent_layout.addRow("纬度最小:", self.lat_min)
        self.extent_layout.addRow("纬度最大:", self.lat_max)
        self.left_layout.addWidget(self.extent_group)

        # --- Display Options ---
        self.display_group = QtWidgets.QGroupBox("显示选项")
        self.display_layout = QtWidgets.QVBoxLayout(self.display_group)

        # Background style
        bg_label = QtWidgets.QLabel("背景颜色:")
        self.bg_layout = QtWidgets.QHBoxLayout()
        self.bg_black = QtWidgets.QRadioButton("黑色")
        self.bg_white = QtWidgets.QRadioButton("白色")
        self.bg_transparent = QtWidgets.QRadioButton("透明")
        self.bg_black.setChecked(True)
        self.bg_layout.addWidget(self.bg_black)
        self.bg_layout.addWidget(self.bg_white)
        self.bg_layout.addWidget(self.bg_transparent)
        self.display_layout.addWidget(bg_label)
        self.display_layout.addLayout(self.bg_layout)

        # Checkboxes
        self.chk_city = QtWidgets.QCheckBox("城市标注")
        self.chk_city.setChecked(True)
        self.chk_shps = QtWidgets.QCheckBox("边界线")
        self.chk_shps.setChecked(True)
        self.chk_grid = QtWidgets.QCheckBox("经纬度网格")
        self.chk_rings = QtWidgets.QCheckBox("距离圈")
        self.chk_labels = QtWidgets.QCheckBox("色标文字")
        self.chk_labels.setChecked(True)
        self.display_layout.addWidget(self.chk_city)
        self.display_layout.addWidget(self.chk_shps)
        self.display_layout.addWidget(self.chk_grid)
        self.display_layout.addWidget(self.chk_rings)
        self.display_layout.addWidget(self.chk_labels)
        self.left_layout.addWidget(self.display_group)

        # --- Custom Shapefile ---
        self.shp_group = QtWidgets.QGroupBox("自定义边界 (可选)")
        self.shp_layout = QtWidgets.QVBoxLayout(self.shp_group)
        self.chk_custom_shp = QtWidgets.QCheckBox("加载自定义SHP")
        self.shp_path = QtWidgets.QLineEdit()
        self.shp_path.setPlaceholderText("shp文件路径")
        self.shp_browse = QtWidgets.QPushButton("浏览...")
        self.shp_row = QtWidgets.QHBoxLayout()
        self.shp_row.addWidget(self.shp_path)
        self.shp_row.addWidget(self.shp_browse)
        self.shp_layout.addWidget(self.chk_custom_shp)
        self.shp_layout.addLayout(self.shp_row)
        self.left_layout.addWidget(self.shp_group)

        self.left_layout.addStretch()

        self.scrollArea.setWidget(self.scrollContent)
        self.left_container_layout.addWidget(self.scrollArea)

        # --- Bottom Bar (fixed, outside scroll) ---
        self.bottom_bar = QtWidgets.QWidget(self.left_container)
        self.bottom_bar.setObjectName("bottom_bar")
        self.bottom_layout = QtWidgets.QVBoxLayout(self.bottom_bar)
        self.bottom_layout.setContentsMargins(6, 2, 6, 2)
        self.bottom_layout.setSpacing(4)

        self.btnSave = QtWidgets.QPushButton("保存图片")
        self.btnSave.setMinimumHeight(32)
        self.bottom_layout.addWidget(self.btnSave)

        self.pushButton_2 = QtWidgets.QPushButton("绘  制")
        self.pushButton_2.setMinimumHeight(50)
        font = self.pushButton_2.font()
        font.setPointSize(12)
        font.setBold(True)
        self.pushButton_2.setFont(font)
        self.bottom_layout.addWidget(self.pushButton_2)

        self.left_container_layout.addWidget(self.bottom_bar)
        self.main_layout.addWidget(self.left_container)

        # ===== Center Panel (Graphics View) =====
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))
        self.main_layout.addWidget(self.graphicsView, 1)

        # ===== Right Panel (File List) =====
        self.right_panel = QtWidgets.QWidget(self.centralwidget)
        self.right_panel.setObjectName("right_panel")
        self.right_panel.setMinimumWidth(180)
        self.right_panel.setMaximumWidth(300)
        self.right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(2, 2, 2, 2)
        self.right_layout.setSpacing(2)

        # --- File Operations ---
        self.file_group = QtWidgets.QGroupBox("文件操作")
        self.file_layout = QtWidgets.QHBoxLayout(self.file_group)
        self.actionOpen = QtWidgets.QPushButton("打开目录")
        self.actionClose = QtWidgets.QPushButton("关闭")
        self.file_layout.addWidget(self.actionOpen)
        self.file_layout.addWidget(self.actionClose)
        self.right_layout.addWidget(self.file_group)

        self.file_list_label = QtWidgets.QLabel("文件列表 (双击加载)")
        self.right_layout.addWidget(self.file_list_label)

        self.file_list = QtWidgets.QListWidget()
        self.file_list.setObjectName("file_list")
        self.right_layout.addWidget(self.file_list)

        self.main_layout.addWidget(self.right_panel)

        MainWindow.setCentralWidget(self.centralwidget)

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PyCINRAD 雷达数据显示 - 增强版"))
