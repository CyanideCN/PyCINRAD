import os
import sys
import ast
import datetime
import traceback
from functools import partial

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from PyQt5 import QtWidgets, QtCore, QtGui
import cinrad
from ui_struct import Ui_MainWindow

# Full product list for base data
BASE_PRODUCTS = [
    'REF', 'TREF', 'VEL', 'SW', 'ZDR', 'RHO', 'PHI', 'KDP',
    'SQI', 'SNRH', 'SNRV', 'LDR', 'CPA', 'CP', 'CF',
    'HCL', 'Zc', 'Vc', 'Wc', 'ZDRc', 'VELSZ', 'POTS', 'COP', 'DR'
]

# Product display names
PROD_NAMES = {
    'REF': '反射率 REF', 'TREF': '总反射率 TREF', 'VEL': '径向速度 VEL',
    'SW': '谱宽 SW', 'ZDR': '差分反射率 ZDR', 'RHO': '相关系数 CC/RHO',
    'PHI': '差分相移 PHI', 'KDP': '差分相移率 KDP', 'SQI': '信号质量指数 SQI',
    'SNRH': '水平信噪比 SNRH', 'SNRV': '垂直信噪比 SNRV',
    'LDR': '退偏振比 LDR', 'CPA': '杂波相位一致性 CPA', 'CP': '杂波可能性 CP',
    'CF': '杂波标志 CF', 'HCL': '水凝物分类 HCL', 'RF': '退折叠标志 RF',
    'Zc': '订正反射率 Zc', 'Vc': '订正速度 Vc', 'Wc': '订正谱宽 Wc',
    'ZDRc': '订正差分反射率 ZDRc', 'VELSZ': 'SZ恢复速度 VELSZ',
    'POTS': '时序相位 POTS', 'COP': '时序相位变化 COP', 'DR': '退极化率 DR',
    'CR': '组合反射率 CR', 'ET': '回波顶高 ET', 'VIL': '垂直积分液态水 VIL',
    'MAX': '最大反射率 MAX', 'OHP': '一小时降水 OHP',
    'CAPPI': '等高面 CAPPI', 'WER': '回波弱回波区 WER',
}

CALC_MAP = {
    1: 'CR', 2: 'ET', 3: 'VIL', 4: 'VILD', 5: 'HCL',
}


# Reader interface map (mirrors cinrad.io.read_auto docstring)
READER_MAP = {
    'read_auto': cinrad.io.read_auto,
    'StandardData': cinrad.io.StandardData,
    'StandardPUP': cinrad.io.StandardPUP,
    'MocMosaic': cinrad.io.MocMosaic,
    'SWAN': cinrad.io.SWAN,
    'CinradReader': cinrad.io.CinradReader,
    'PhasedArrayData': cinrad.io.PhasedArrayData,
}


def _parse_reader_args(args_str):
    """Parse extra kwargs string like 'radar_type=SA, foo=1' into a dict."""
    kwargs = {}
    if not args_str or not args_str.strip():
        return kwargs
    for part in args_str.split(','):
        part = part.strip()
        if not part or '=' not in part:
            continue
        k, v = part.split('=', 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            kwargs[k] = ast.literal_eval(v)
        except Exception:
            kwargs[k] = v
    return kwargs


def read(fpath, reader='read_auto', args_str=''):
    """Read radar data with the selected interface.

    Args:
        fpath: path of radar data file.
        reader: interface name in READER_MAP, default 'read_auto'.
        args_str: extra keyword arguments string, e.g. 'radar_type=SA'.
    """
    try:
        kwargs = _parse_reader_args(args_str)
        if reader in READER_MAP and reader != 'read_auto':
            return READER_MAP[reader](fpath, **kwargs)
        return cinrad.io.read_auto(fpath)
    except Exception:
        try:
            return cinrad.io.read_auto(fpath)
        except Exception:
            try:
                return cinrad.io.CinradReader(fpath)
            except Exception:
                try:
                    return cinrad.io.StandardData(fpath)
                except Exception:
                    return None


class Figure_Canvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)


class DrawWorker(QtCore.QObject):
    """Worker for non-blocking drawing in a separate thread."""
    finished = QtCore.pyqtSignal(object)   # emits the PPI fig_obj or None
    error = QtCore.pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f'{e}\n{traceback.format_exc()}')


class RadarUI(Ui_MainWindow):
    def __init__(self):
        self.last_fname = None
        self.tilt = 0
        self.dtype = 'REF'
        self.drange = 230
        self.redraw = False
        self._fig_obj = None

    def setupUi(self, MainWindow):
        super(RadarUI, self).setupUi(MainWindow)
        self.main_window = MainWindow
        self.graphicscene = QtWidgets.QGraphicsScene()

        # GraphicsView: enable smooth zoom & drag
        self.graphicsView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.graphicsView.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.graphicsView.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.graphicsView.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.graphicsView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.graphicsView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.graphicsView.wheelEvent = self._graphics_view_wheel_event

        # Connect signals
        self.actionOpen.clicked.connect(self._open_dir)
        self.actionClose.clicked.connect(self._close)
        self.pushButton_2.clicked.connect(self.draw)
        self.pushButton.clicked.connect(self.on_range_update)
        self.comboBox.activated.connect(self.on_combobox_activate)
        self.comboBox.currentIndexChanged.connect(self.on_combobox_changed)
        self.shp_browse.clicked.connect(self._browse_shp)
        self.btnSave.clicked.connect(self._save)
        self.chk_rings.stateChanged.connect(self._on_rings_toggle)
        self.file_list.itemDoubleClicked.connect(self._load_selected_file)
        # Height combo change triggers redraw if image exists
        self.height_combo.currentIndexChanged.connect(self._on_height_changed)

        # Range rings group
        self.rings_widget = QtWidgets.QWidget()
        rings_layout = QtWidgets.QHBoxLayout(self.rings_widget)
        rings_layout.setContentsMargins(20, 0, 0, 0)
        self.rings_label = QtWidgets.QLabel("圈半径(km,逗号分隔):")
        self.rings_input = QtWidgets.QLineEdit("50,100,150,200")
        self.rings_input.setMaximumWidth(180)
        rings_layout.addWidget(self.rings_label)
        rings_layout.addWidget(self.rings_input)
        self.display_layout.addWidget(self.rings_widget)
        self.rings_widget.setVisible(False)

        # Populate product combo
        self._populate_products()

        # Background radio buttons - use QButtonGroup to avoid double-trigger
        self.bg_group = QtWidgets.QButtonGroup(self.main_window)
        self.bg_group.addButton(self.bg_black)
        self.bg_group.addButton(self.bg_white)
        self.bg_group.addButton(self.bg_transparent)
        self.bg_group.buttonClicked.connect(self._on_bg_changed)

        # Initial state
        self.height_group.setVisible(False)
        self._set_controls_enabled(False)
        self._drawing = False

    def _populate_products(self):
        self.prod_combo.clear()
        for p in BASE_PRODUCTS:
            name = PROD_NAMES.get(p, p)
            self.prod_combo.addItem(name, p)

    def _set_controls_enabled(self, enabled):
        self.comboBox.setEnabled(enabled)
        self.prod_combo.setEnabled(enabled)
        self.calc_combo.setEnabled(enabled)
        self.pushButton_2.setEnabled(enabled)

    def _on_bg_changed(self, button):
        """When background changes: update view background, and if image exists, redraw."""
        if self.bg_black.isChecked():
            self.graphicsView.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))
        elif self.bg_white.isChecked():
            self.graphicsView.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.white))
        else:
            self.graphicsView.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.transparent))
        # If an image is already drawn, redraw with new style
        if self.redraw and not self._drawing and hasattr(self, 'cinrad') and self.cinrad is not None:
            self.draw()

    def _graphics_view_wheel_event(self, event):
        """Mouse wheel zoom for QGraphicsView."""
        if event.modifiers() & QtCore.Qt.ControlModifier:
            zoom_factor = 1.15
        else:
            zoom_factor = 1.05
        if event.angleDelta().y() > 0:
            self.graphicsView.scale(zoom_factor, zoom_factor)
        else:
            self.graphicsView.scale(1.0 / zoom_factor, 1.0 / zoom_factor)

    def _on_rings_toggle(self, state):
        self.rings_widget.setVisible(state == QtCore.Qt.Checked)

    def _browse_shp(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window, "选择Shapefile", "", "Shapefile (*.shp)")
        if fname:
            self.shp_path.setText(fname)

    def _get_style(self):
        if self.bg_black.isChecked():
            return 'black'
        elif self.bg_white.isChecked():
            return 'white'
        else:
            return 'transparent'

    def _get_extent(self):
        if not self.extent_check.isChecked():
            return None
        try:
            lon_min = float(self.lon_min.text())
            lon_max = float(self.lon_max.text())
            lat_min = float(self.lat_min.text())
            lat_max = float(self.lat_max.text())
            return [lon_min, lon_max, lat_min, lat_max]
        except (ValueError, AttributeError):
            return None

    def _open_dir(self):
        """Open a directory and populate the file list."""
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self.main_window, "打开雷达数据目录", "")
        if not d:
            return
        self._current_dir = d
        self.file_list.clear()
        # Supported radar file extensions
        exts = ('.bin', '.bz2', '.bzip2', '.dat', '.raw', '.nc', '.gz')
        try:
            files = sorted([f for f in os.listdir(d)
                           if f.lower().endswith(exts)])
        except Exception:
            files = []
        if not files:
            self._message('该目录下未找到雷达数据文件')
            return
        for f in files:
            self.file_list.addItem(f)
        self.statusbar.showMessage(f'目录已打开: {d} ({len(files)} 个文件)', 5000)

    def _load_selected_file(self, item):
        """Load the double-clicked file from the file list."""
        if item is None:
            return
        fn = os.path.join(getattr(self, '_current_dir', ''), item.text())
        if not os.path.exists(fn):
            return

        # Show loading status. read() below is blocking, so force a repaint
        # first, otherwise the message would only appear after loading ends.
        self.statusbar.showMessage('正在加载，请稍后...', 0)
        QtWidgets.QApplication.processEvents()

        try:
            reader = self.reader_combo.currentData()
            if not reader:
                reader = 'read_auto'
            args = self.reader_args.text().strip()
            self.cinrad = read(fn, reader, args)
        except Exception:
            self.cinrad = None

        if self.cinrad is None:
            self.statusbar.clearMessage()
            self._message('无法读取该数据，请检查文件格式')
            return

        self.last_fname = fn

        # Flush old state
        self._flush()

        # Display basic info
        name = getattr(self.cinrad, 'name', '未知')
        stime = getattr(self.cinrad, 'scantime', None)
        if stime:
            if hasattr(stime, 'strftime'):
                time_str = stime.strftime('%Y-%m-%d %H:%M:%S UTC')
            else:
                time_str = str(stime)
        else:
            time_str = '未知'
        code = getattr(self.cinrad, 'code', '')
        lon = getattr(self.cinrad, 'stationlon', None)
        lat = getattr(self.cinrad, 'stationlat', None)
        lon_str = f'{lon:.4f}' if lon is not None else '未知'
        lat_str = f'{lat:.4f}' if lat is not None else '未知'

        info = '站名: {}\n站号: {}\n扫描时间: {}\n经度: {}\n纬度: {}'.format(
            name, code, time_str, lon_str, lat_str)
        self.label_3.setText(info)

        # Status bar message
        reader_type = type(self.cinrad).__name__
        self.statusbar.showMessage(f'已加载: {os.path.basename(fn)} ({reader_type})')

        # Determine data type and populate tilt/product
        self._populate_tilts()
        self._set_controls_enabled(True)

        # Only base data can calculate products
        self._update_calc_visibility()

        # Auto draw once the file is loaded
        self.draw()

    def _populate_tilts(self):
        self.comboBox.clear()
        el = getattr(self.cinrad, 'el', None)

        if el is not None and hasattr(el, '__iter__'):
            # Base data with tilts
            for i, e in enumerate(el):
                try:
                    self.comboBox.addItem('仰角{}-{:.2f}deg'.format(i, float(e)))
                except (ValueError, TypeError):
                    self.comboBox.addItem('仰角{}'.format(i))
            self.tilt_group.setVisible(True)
        else:
            # PUP/SWAN/MocMosaic - no tilt selection
            self.comboBox.addItem('N/A')
            self.tilt_group.setVisible(False)
            self.tilt = 0

        # Check for height dimension (3D data like CAPPI)
        self._check_height_dimension()

    def _check_height_dimension(self):
        """Detect if data has a height dimension and populate height combo."""
        self.height_combo.clear()
        heights = None

        # Possible coordinate names for height levels
        height_coords = ['height', 'level', 'altitude', 'z']

        try:
            if hasattr(self.cinrad, 'el') and hasattr(self.cinrad, 'get_data'):
                # Base data: try all available products to find one with height dim
                if hasattr(self.cinrad, 'available_product'):
                    ap = self.cinrad.available_product(self.tilt)
                    print(f'[height] available_product({self.tilt}): {ap}')
                    # Try common products first, then all available
                    check_list = [p for p in ['REF', 'VEL', 'SW', 'ZDR', 'KDP']
                                  if p in ap] + list(ap)
                    for prod in check_list:
                        try:
                            test_data = self.cinrad.get_data(self.tilt, self.drange, prod)
                            if test_data is None:
                                continue
                            # Check for height-like coordinate
                            for hc in height_coords:
                                if hc in test_data.dims or hc in test_data.coords:
                                    hvals = test_data.coords.get(hc)
                                    if hvals is not None and len(hvals) > 1:
                                        heights = hvals.values
                                        print(f'[height] Found heights via {prod}.{hc}: {heights}')
                                        break
                            if heights is not None:
                                break
                        except Exception as ex:
                            print(f'[height] get_data({prod}) error: {ex}')
                            continue
                else:
                    # Base data without available_product - try common products
                    for prod in ['REF', 'VEL', 'SW']:
                        try:
                            test_data = self.cinrad.get_data(self.tilt, self.drange, prod)
                            if test_data is None:
                                continue
                            for hc in height_coords:
                                if hc in test_data.dims or hc in test_data.coords:
                                    hvals = test_data.coords.get(hc)
                                    if hvals is not None and len(hvals) > 1:
                                        heights = hvals.values
                                        print(f'[height] Found heights via {prod}.{hc}: {heights}')
                                        break
                            if heights is not None:
                                break
                        except Exception as ex:
                            print(f'[height] get_data({prod}) error: {ex}')
                            continue

            elif hasattr(self.cinrad, 'get_data'):
                # PUP/SWAN/MocMosaic - get_data() returns all products
                test_data = self.cinrad.get_data()
                if test_data is not None:
                    print(f'[height] PUP data vars: {list(test_data.data_vars)}')
                    print(f'[height] PUP dims: {dict(test_data.dims)}')
                    print(f'[height] PUP coords: {list(test_data.coords)}')
                    for hc in height_coords:
                        if hc in test_data.dims or hc in test_data.coords:
                            hvals = test_data.coords.get(hc)
                            if hvals is not None and len(hvals) > 1:
                                heights = hvals.values
                                print(f'[height] Found heights via {hc}: {heights}')
                                break

                    # Also check if CAPPI variable exists with height
                    if heights is None and 'CAPPI' in test_data.data_vars:
                        cappi_data = test_data['CAPPI']
                        for hc in height_coords:
                            if hc in cappi_data.dims or hc in cappi_data.coords:
                                hvals = cappi_data.coords.get(hc)
                                if hvals is not None and len(hvals) > 1:
                                    heights = hvals.values
                                    print(f'[height] Found CAPPI heights via {hc}: {heights}')
                                    break

        except Exception as e:
            print(f'[height] _check_height_dimension error: {e}')

        if heights is not None and len(heights) > 1:
            for h in heights:
                self.height_combo.addItem('{:.1f}'.format(float(h)), float(h))
            self.height_combo.setCurrentIndex(0)
            self.height_group.setVisible(True)
            self.statusbar.showMessage(f'检测到 {len(heights)} 个高度层', 3000)
        else:
            self.height_group.setVisible(False)

    def _get_selected_height(self):
        """Get the selected height value, or None if not applicable."""
        if not self.height_group.isVisible():
            return None
        idx = self.height_combo.currentIndex()
        if idx < 0:
            return None
        return self.height_combo.itemData(idx)

    def _on_height_changed(self):
        """Redraw when height selection changes."""
        if self.redraw and not self._drawing:
            self.draw()

    def _update_calc_visibility(self):
        """Only base data (with el attribute) can select/calculate products."""
        is_base = hasattr(self.cinrad, 'el') and hasattr(self.cinrad, 'available_product')
        # Hide product selection for PUP/SWAN/MocMosaic
        self.prod_group.setVisible(is_base)
        # Calc products only for base data
        self.calc_label.setVisible(is_base)
        self.calc_combo.setVisible(is_base)
        if not is_base:
            self.calc_combo.setCurrentIndex(0)

    def _flush(self):
        self.comboBox.clear()
        self.height_combo.clear()
        self.height_group.setVisible(False)
        self.label_3.setText('')
        self.graphicscene.clear()
        if hasattr(self, 'graphicsView'):
            self.graphicsView.setScene(self.graphicscene)

    def on_combobox_activate(self, index):
        self.tilt = index
        self.on_combobox_changed()

    def on_combobox_changed(self):
        """Update available products based on selected tilt"""
        if not hasattr(self, 'cinrad'):
            return
        if self.cinrad is None:
            return

        # For base data, check available products
        if hasattr(self.cinrad, 'available_product'):
            try:
                ap = self.cinrad.available_product(self.tilt)
                # Update product combo to show only available products
                current_prod = self.prod_combo.currentData()
                self.prod_combo.clear()
                for p in BASE_PRODUCTS:
                    if p in ap:
                        name = PROD_NAMES.get(p, p)
                        self.prod_combo.addItem(name, p)
                # Try to restore selection
                if current_prod:
                    idx = self.prod_combo.findData(current_prod)
                    if idx >= 0:
                        self.prod_combo.setCurrentIndex(idx)
            except Exception:
                pass

    def on_range_update(self):
        rad = self.plainTextEdit.text()
        try:
            self.drange = float(rad)
            self.statusbar.showMessage(f'绘图半径已更新: {self.drange} km', 3000)
        except ValueError:
            self._message('请输入有效的数字')

    def _get_dpi(self):
        try:
            return int(self.dpi_input.text())
        except (ValueError, AttributeError):
            return 300

    def _process_data(self, data):
        """Process data for common variable name issues"""
        if data is None:
            return None
        # Rename common PUP variable names
        rename_map = {
            'Zc': 'REF', 'Vc': 'VEL', 'Wc': 'SW', 'ZDRc': 'ZDR',
            'VELSZ': 'VEL',
        }
        for old_name, new_name in rename_map.items():
            if old_name in data.data_vars and new_name not in data.data_vars:
                data = data.rename({old_name: new_name})

        # Handle CAPPI (has height dimension)
        if 'CAPPI' in data.data_vars:
            if 'height' in data.dims:
                h = self._get_selected_height()
                if h is not None:
                    data = data.sel(height=h)
                else:
                    data = data.sel(height=data.coords['height'][0])
            data = data.rename({'CAPPI': 'REF'})

        # Handle any variable with height dimension - let user pick height
        if 'height' in data.dims:
            h = self._get_selected_height()
            if h is not None:
                data = data.sel(height=h)
            else:
                data = data.sel(height=data.coords['height'][0])

        # Handle WER (multiple height layers)
        wer_vars = [v for v in data.data_vars if v.startswith('WER_')]
        if wer_vars:
            # Select first available WER layer
            data = data.rename({wer_vars[0]: 'REF'})

        # Mask negative values for some products
        if self.dtype in ('REF', 'TREF', 'CR', 'MAX'):
            import numpy as np
            if self.dtype in data.data_vars:
                vals = data[self.dtype].values
                if hasattr(vals, 'filled'):
                    data[self.dtype].values = np.ma.masked_less(vals, 0)

        return data

    def _trim_to_extent(self, data, extent):
        """Crop data to the user-defined lon/lat extent for faster rendering."""
        if not extent or data is None:
            return data
        try:
            lon_min, lon_max, lat_min, lat_max = extent
            lon = data['longitude']
            lat = data['latitude']
            cropped = data.where(
                (lon >= lon_min) & (lon <= lon_max)
                & (lat >= lat_min) & (lat <= lat_max),
                drop=True,
            )
            # Fall back to original data if cropping leaves it empty
            # (i.e. the extent does not overlap the data).
            if all(v > 0 for v in cropped.sizes.values()):
                return cropped
        except Exception as e:
            print(f'[trim] extent crop skipped: {e}')
        return data

    def draw(self):
        if self._drawing:
            self.statusbar.showMessage('正在绘制中，请稍候...', 2000)
            return

        if self.redraw:
            plt.close('all')
            self.graphicscene.clear()

        calc_idx = self.calc_combo.currentIndex()
        style = self._get_style()
        dpi = self._get_dpi()
        extent = self._get_extent()
        add_city = self.chk_city.isChecked()
        add_shps = self.chk_shps.isChecked()
        plot_labels = self.chk_labels.isChecked()
        use_grid = self.chk_grid.isChecked()
        use_rings = self.chk_rings.isChecked()
        use_custom_shp = self.chk_custom_shp.isChecked()
        shp_path = self.shp_path.text().strip()

        # Determine text color based on style
        if style == 'white':
            text_color = 'black'
        else:
            text_color = 'white'

        try:
            # Check if this is a calculated product
            if calc_idx > 0:
                data = self._calc_product(calc_idx)
            elif hasattr(self.cinrad, 'el') or hasattr(self.cinrad, 'get_data'):
                # Base data - get_data needs tilt, range, dtype
                self.dtype = self.prod_combo.currentData()
                if self.dtype is None:
                    self.dtype = 'REF'
                try:
                    data = self.cinrad.get_data(self.tilt, self.drange, self.dtype)
                except Exception as e:
                    # Some PUP files don't need tilt/range
                    try:
                        data = self.cinrad.get_data()
                    except Exception as e2:
                        self._message(f'读取数据失败: {e2}')
                        return
            else:
                # PUP/SWAN/MocMosaic - no tilt needed
                try:
                    data = self.cinrad.get_data()
                except Exception as e:
                    self._message(f'读取数据失败: {e}')
                    return

            if data is None:
                self._message('无法获取数据')
                return

            # Process data for display
            data = self._process_data(data)
            if data is None:
                self._message('数据处理失败')
                return

            # Crop data to the user-defined extent to speed up rendering
            data = self._trim_to_extent(data, extent)

            # --- Data ready, now run PPI construction in a thread ---
            # Capture all UI values before starting thread (avoid cross-thread access)
            rings_text = self.rings_input.text().strip() if use_rings else ''

            self._drawing = True
            self.pushButton_2.setEnabled(False)
            self.pushButton_2.setText('绘制中...')
            self.statusbar.showMessage('正在绘制...', 0)

            def _do_draw():
                # Create figure canvas - use fixed display DPI for performance
                dr = Figure_Canvas(dpi=100)

                # Set figure background based on style
                if style != 'transparent':
                    dr.figure.patch.set_facecolor(style)
                else:
                    dr.figure.patch.set_alpha(0)

                # Build PPI
                kwargs = dict(
                    fig=dr.figure,
                    style=style,
                    add_city_names=add_city,
                    add_shps=add_shps,
                    plot_labels=plot_labels,
                    dpi=100,
                    text_param={"color": text_color, "fontsize": 10},
                )
                if extent:
                    kwargs['extent'] = extent

                fig_obj = cinrad.visualize.PPI(data, **kwargs)

                # In PyQt (non-inline backend), PPI skips drawing the colorbar
                # and info text at init (it only does that in _save()). Since we
                # don't use PPI's save path, trigger it manually here.
                if not fig_obj.settings.get("is_inline", False):
                    try:
                        fig_obj._text_before_save()
                    except Exception as e:
                        print(f'Text/colorbar rendering error: {e}')

                # Add grid lines
                if use_grid:
                    try:
                        fig_obj.gridlines(draw_labels=True, linewidth=0.5, color=text_color)
                    except Exception as e:
                        print(f'Grid lines error: {e}')

                # Add range rings
                if use_rings and rings_text:
                    try:
                        rings = [float(x.strip()) for x in rings_text.split(',')]
                        fig_obj.plot_range_rings(rings, color=text_color, linewidth=1)
                    except Exception as e:
                        print(f'Range rings error: {e}')

                # Add custom shapefile
                if use_custom_shp and shp_path:
                    if os.path.exists(shp_path):
                        try:
                            fig_obj.add_custom_shp(shp_path, encoding="gbk",
                                                   color=text_color, linewidth=1)
                        except Exception as e:
                            print(f'Custom shapefile error: {e}')

                return fig_obj, dr

            # Store params for completion handler
            self._save_dpi = dpi

            # Set up worker thread
            self._worker = DrawWorker(_do_draw)
            self._worker_thread = QtCore.QThread()
            self._worker.moveToThread(self._worker_thread)
            self._worker.finished.connect(self._on_draw_finished)
            self._worker.error.connect(self._on_draw_error)
            self._worker_thread.started.connect(self._worker.run)
            self._worker_thread.start()

        except Exception as e:
            traceback.print_exc()
            self._drawing = False
            self.pushButton_2.setEnabled(True)
            self.pushButton_2.setText('绘  制')
            self._message(f'绘制失败: {e}')

    def _on_draw_finished(self, result):
        """Called when drawing thread completes - runs in main thread."""
        self._worker_thread.quit()
        if result is None:
            self._on_draw_error('绘制返回空结果')
            return
        fig_obj, dr = result
        self._fig_obj = fig_obj
        # Add to graphics view (must be in main thread)
        self.graphicscene.addWidget(dr)
        self.graphicsView.setScene(self.graphicscene)
        self.graphicsView.show()
        QtCore.QTimer.singleShot(50, self._fit_to_view)
        self.redraw = True
        self._drawing = False
        self.pushButton_2.setEnabled(True)
        self.pushButton_2.setText('绘  制')
        self.statusbar.showMessage('绘制完成', 3000)

    def _on_draw_error(self, msg):
        """Called when drawing thread fails."""
        self._worker_thread.quit()
        self._drawing = False
        self.pushButton_2.setEnabled(True)
        self.pushButton_2.setText('绘  制')
        traceback.print_exc()
        self._message(f'绘制失败: {msg}')

    def _calc_product(self, calc_idx):
        """Calculate derived products"""
        prod = CALC_MAP.get(calc_idx)
        if not prod:
            return None

        try:
            if prod == 'HCL':
                # HCL needs REF, ZDR, RHO, KDP from same tilt
                ref = self.cinrad.get_data(self.tilt, self.drange, 'REF')
                zdr = self.cinrad.get_data(self.tilt, self.drange, 'ZDR')
                rho = self.cinrad.get_data(self.tilt, self.drange, 'RHO')
                kdp = self.cinrad.get_data(self.tilt, self.drange, 'KDP')
                result = cinrad.calc.hydro_class(ref, zdr, rho, kdp, band="S")
                self.dtype = 'HCL'
                return result
            else:
                # CR, ET, VIL, VILD need all tilts
                r_list = list(self.cinrad.iter_tilt(self.drange, 'REF'))
                if prod == 'CR':
                    result = cinrad.calc.quick_cr(r_list)
                elif prod == 'ET':
                    result = cinrad.calc.quick_et(r_list)
                elif prod == 'VIL':
                    result = cinrad.calc.quick_vil(r_list)
                elif prod == 'VILD':
                    result = cinrad.calc.quick_vild(r_list)
                else:
                    return None
                self.dtype = prod
                return result
        except Exception as e:
            self._message(f'计算产品失败: {e}')
            return None

    def _fit_to_view(self):
        """Scale image to fit the graphics view viewport."""
        rect = self.graphicscene.sceneRect()
        if rect.width() > 0 and rect.height() > 0:
            self.graphicsView.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def _save(self):
        if self._fig_obj is None:
            self._message('请先绘制图形')
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.main_window, "保存图片", "",
            "PNG (*.png);;JPG (*.jpg);;所有文件 (*)")
        if fname:
            try:
                # Set user-specified DPI for saving
                save_dpi = getattr(self, '_save_dpi', 300)
                self._fig_obj.fig.set_dpi(save_dpi)
                # The colorbar/text were already drawn manually at plot time.
                # PPI._save() would draw them again when is_inline is False,
                # so temporarily flag it to avoid duplicated overlays.
                orig_inline = self._fig_obj.settings.get("is_inline", False)
                self._fig_obj.settings["is_inline"] = True
                try:
                    self._fig_obj(fname)
                finally:
                    self._fig_obj.settings["is_inline"] = orig_inline
                # Restore display DPI
                self._fig_obj.fig.set_dpi(100)
                self.statusbar.showMessage(f'图片已保存: {fname}', 5000)
            except Exception as e:
                self._message(f'保存失败: {e}')

    def _close(self):
        plt.close('all')
        self.redraw = False
        self._fig_obj = None
        self.file_list.clear()
        if hasattr(self, 'cinrad'):
            del self.cinrad
        self._flush()
        self._set_controls_enabled(False)
        self.calc_label.setVisible(True)
        self.calc_combo.setVisible(True)
        self.statusbar.showMessage('已关闭数据', 3000)

    def _message(self, message):
        msg = QtWidgets.QMessageBox.warning(self.main_window, '提示', message)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = RadarUI()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
