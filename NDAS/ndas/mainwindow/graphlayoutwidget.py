import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5.QtWidgets import QApplication, QGraphicsRectItem
from pyqtgraph.Qt import QtCore
from PyQt5.QtGui import *

from ndas.extensions import algorithms, annotations
from ndas.misc.colors import Color


class GraphLayoutWidget(pg.GraphicsLayoutWidget):
    """
    Widget for the main plot in annotation view
    """
    mouse_moved_signal = QtCore.pyqtSignal(float, float)
    fps_updated_signal = QtCore.pyqtSignal(int)
    point_selection_changed_signal = QtCore.pyqtSignal(int)
    point_labeling_changed_signal = QtCore.pyqtSignal(int)

    _x_data = None
    _y_data = None

    _v_line = None
    _v_line_enabled = False
    _h_line = None
    _h_line_enabled = False

    main_plot = None
    nav_plot = None

    main_plot_name = None
    main_dot_plot_item = None
    main_line_plot_item = None
    nav_plot_item = None

    supplementary_plots = []

    _mouse_pointer = {"x": 0, "y": 0}
    _selected_pointer = []

    _plot_background_color = 'eeeeee'

    toolTipFlag = True
    showLabels = True

    def __init__(self, *args, **kwargs):
        """
        Initializes the nav plot and the main plot.
        Initializes the moused move signal.

        Parameters
        ----------
        args
        kwargs
        """
        super(GraphLayoutWidget, self).__init__(*args, **kwargs)

        self._init_main_plot(1, 0)
        self._init_nav_plot(2, 0)
        self._init_fps_timer()

        self._mouse_moved_proxy = pg.SignalProxy(self.main_plot.scene().sigMouseMoved, rateLimit=60,
                                                 slot=self._mouse_moved_event)

    def _init_main_plot(self, layout_row: int, layout_col: int):
        """
        Loads the main plot.

        Parameters
        ----------
        layout_row
        layout_col
        """
        self.main_plot = self.addPlot(row=layout_row, col=layout_col, viewBox=MultiSelectViewBox())
        # self.main_plot.setMenuEnabled(False)
        self.main_plot.setAutoVisible(y=True)
        self.main_plot.showGrid(x=True, y=True, alpha=.3)
        self.main_plot.getViewBox().multipoint_selection_changed_signal.connect(lambda val: self._multiselect(val))

        self.drag_box = QGraphicsRectItem(0, 0, 1, 1)
        self.drag_box.setPen(pg.mkPen(self.palette().color(QPalette.Highlight), width=1))
        self.drag_box.setBrush(pg.mkBrush(self.palette().color(QPalette.Highlight).getRgb()[:3] + (64,)))
        self.drag_box.setZValue(1e9)
        self.drag_box.hide()
        self.main_plot.addItem(self.drag_box, ignoreBounds=True)

        self._h_line = pg.InfiniteLine(angle=0, movable=False, pen={'color': "FF0000", 'alpha': 0.3})
        self._v_line = pg.InfiniteLine(angle=90, movable=False, pen={'color': "FF0000", 'alpha': 0.3})

    def _init_nav_plot(self, layout_row: int, layout_col: int):
        """
        Loads the navigation plot.

        Parameters
        ----------
        layout_row
        layout_col
        """
        self.nav_plot = self.addPlot(row=layout_row, col=layout_col)
        self.nav_plot.setFixedHeight(self.height() * 0.20)
        self.nav_plot.setMenuEnabled(False)
        self.nav_plot.setMouseEnabled(False, False)

    def _update_nav_plot(self):
        """
        Updates the nav plot if the main plot is moved
        """
        self.region = pg.LinearRegionItem(brush=(225, 225, 255, 60), pen=({'color': "FF0000", 'width': 3}))
        self.region.setZValue(10)
        self.nav_plot.addItem(self.region, ignoreBounds=True)
        self.region.sigRegionChanged.connect(self._update_plot_region)
        self.main_plot.sigRangeChanged.connect(self._update_region)
        self.region.setRegion([1000, 2000])

    def _init_fps_timer(self):
        """
        Initializes the fps timer
        """
        self.last_update = pg.ptime.time()
        self.avg_fps = 0.0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_fps)
        self.timer.start(100)

    def _update_plot_region(self):
        """
        Updates the region for the navigation plot
        (if changes in navigation plot)
        """
        self.region.setZValue(10)
        min_x, max_x = self.region.getRegion()
        self.main_plot.setXRange(min_x, max_x, padding=0)

    def _update_region(self, window, view_range):
        """
        Updates the region of the main plot
        (if changed in main plot)

        Parameters
        ----------
        window
        view_range
        """
        rgn = view_range[0]
        self.region.setRegion(rgn)

    def _update_fps(self):
        """
        Calculates the fps and emits the signal
        """
        fps = 100.0 / (pg.ptime.time() - self.last_update)
        self.last_update = pg.ptime.time()
        self.avg_fps = self.avg_fps * 0.8 + fps * 0.2
        self.fps_updated_signal.emit(self.avg_fps)

    def set_h_line_visibility(self, status):
        """
        Changes the horizontal line visibility

        Parameters
        ----------
        status
        """
        if status:
            _h_line_enabled = True
            self.main_plot.addItem(self._h_line, ignoreBounds=True)
        else:
            _h_line_enabled = False
            self.main_plot.removeItem(self._h_line)

    def set_v_line_visibility(self, status):
        """
        Changes the vertical line visibility

        Parameters
        ----------
        status
        """
        if status:
            _v_line_enabled = True
            self.main_plot.addItem(self._v_line, ignoreBounds=True)
        else:
            _v_line_enabled = False
            self.main_plot.removeItem(self._v_line)

    def draw_registered_plot(self, registered_plot_item):
        """
        Draws a plot to the main and navigation graph

        Parameters
        ----------
        registered_plot_item
        """
        if self.main_dot_plot_item is not None:
            self.deselect_all()
            self.main_dot_plot_item.disconnect()
            self.main_plot.removeItem(self.main_dot_plot_item)
            self.main_dot_plot_item = None
        if self.main_line_plot_item is not None:
            self.main_plot.removeItem(self.main_line_plot_item)
            self.main_line_plot_item = None
        if self.supplementary_plots:
            for item in self.supplementary_plots:
                self.main_plot.removeItem(item)

        self._x_data = registered_plot_item.main_dot_plot.x_data.to_numpy()
        self._y_data = registered_plot_item.main_dot_plot.y_data.to_numpy()

        self.main_dot_plot_item = registered_plot_item.main_dot_plot.plot_item
        self.main_line_plot_item = registered_plot_item.main_line_plot.plot_item
        self.main_plot_name = registered_plot_item.main_dot_plot.plot_item_name

        if registered_plot_item.dot_plot_visible:
            self.main_plot.addItem(self.main_dot_plot_item)

        if registered_plot_item.line_plot_visible:
            self.main_plot.addItem(self.main_line_plot_item)

        if self._h_line_enabled:
            self.main_plot.addItem(self._h_line, ignoreBounds=True)

        if self._v_line_enabled:
            self.main_plot.addItem(self._v_line, ignoreBounds=True)

        for item in registered_plot_item.supplementary_plots:
            self.supplementary_plots.append(item.plot_item)
            self.main_plot.addItem(item.plot_item, ignoreBounds=True)

        self.main_plot.setLabel(axis='left', text=registered_plot_item.y_label)
        self.main_plot.setLabel(axis='bottom', text=registered_plot_item.x_label)

        self.nav_plot.clear()
        self.nav_plot.plot(y=self._y_data, x=self._x_data, pen=Color.BLUE.value)

        self.nav_plot.setLabel(axis='left', text=registered_plot_item.y_label)
        self.nav_plot.setLabel(axis='bottom', text=registered_plot_item.x_label)

        self.main_dot_plot_item.sigClicked.connect(self._clicked_point_slot)
        self.main_plot.sigRangeChanged.connect(self._update_region)
        self._update_nav_plot()

        self.update_labels()
        self.main_plot.autoRange()

        self.main_dot_plot_item.opts['hoverable'] = True
        self.main_dot_plot_item.sigHovered.connect(self.showTooltip)

    def showTooltip(self, plot, points):
        if len(points) > 0 and self.toolTipFlag == True:
            if len(self.main_plot_name.split("(")) > 1:
                tooltip = "Value: " +str(round(points[0].pos()[1],3)) + " " + (self.main_plot_name.split("(")[1])[:-1]+ "\nTimestamp: " + str(points[0].pos()[0])
            else:
                tooltip = "Value: " +str(round(points[0].pos()[1],3)) + "\nTimestamp: " + str(points[0].pos()[0])
            for point in annotations.get_labeled_points(self.main_plot_name):
                state = point.__getstate__()
                if state[0] == points[0].pos()[1] and state[1] == points[0].pos()[0]:
                    tooltip = tooltip + "\nLabel: " + point.label
            self.main_dot_plot_item.setToolTip(tooltip)
        else: 
            self.main_dot_plot_item.setToolTip(None)

    def toggleTooltipFlag(self, flag):
        self.toolTipFlag = flag

    def toggleLabelFlag(self, flag):
        self.showLabels = flag
        self.update_labels()

    def set_line_item_visibility(self, status):
        """
        Changes the visibility of the connecting lines

        Parameters
        ----------
        status
        """
        if self.main_line_plot_item is not None:
            self.main_plot.removeItem(self.main_line_plot_item)

        if status:
            self.main_plot.addItem(self.main_line_plot_item)

    def set_point_item_visibility(self, status):
        """
        Changes the visibility of the scatter plot items

        Parameters
        ----------
        status
        """
        if self.main_dot_plot_item is not None:
            self.main_plot.removeItem(self.main_dot_plot_item)

        if status:
            self.main_plot.addItem(self.main_dot_plot_item)

    def clear_plots(self):
        """
        Removes currently visible plots from boths plot areas
        """
        self.main_plot.clear()
        self.main_plot.addItem(self.drag_box)
        self.nav_plot.clear()

    def set_background_color(self, pbg: str):
        """
        Sets a specific background color for the graph

        Parameters
        ----------
        pbg
        """
        self._plot_background_color = pbg
        vb = self.main_plot.getViewBox()
        vb.setBackgroundColor(self._plot_background_color)

    def _deselect(self, point: pg.ScatterPlotItem):
        """
        Deselects the given scatter plot item

        Parameters
        ----------
        point
        """
        point.setSymbol('o')
        point.setSize(10)
        point.setPen('ffffff')
        self.point_selection_changed_signal.emit(annotations.get_number_selected())

    def _select(self, point: pg.ScatterPlotItem):
        """
        Selects the given scatter plot item

        Parameters
        ----------
        point
        """
        point.setSymbol('x')
        point.setSize(14)
        point.setPen('17b12d', width=2)
        self.point_selection_changed_signal.emit(annotations.get_number_selected())

    def _multiselect(self, val):
        """
        Selects a selection of scatter plot items at once

        Parameters
        ----------
        val
        """
        bottom_left, top_right = val

        indexes = []
        for i in range(len(self._x_data)):
            x = self._x_data[i]
            y = self._y_data[i]
            if bottom_left[0] <= x <= top_right[0]:
                if bottom_left[1] <= y <= top_right[1]:
                    lx = np.argwhere(self._x_data == x)
                    ly = np.argwhere(self._y_data == y)
                    i = np.intersect1d(lx, ly)[0]
                    e = {'id': i, 'x': x, 'y': y}
                    indexes.append(e)

        spot_items = self.main_dot_plot_item.points()
        for single_index in list({v['id']: v for v in indexes}.values()):
            point = annotations.SelectedPoint(single_index['y'], single_index['x'], single_index['id'])

            if annotations.is_point_selected(point):
                annotations.remove_selected_point(point)
                self._deselect(spot_items[single_index['id']])
            else:
                annotations.add_selection_point(point)
                self._select(spot_items[single_index['id']])

    def _remove_all_labels(self):
        """
        Removes all visible labels from plot
        """
        for child in self.main_plot.getViewBox().allChildren():
            if isinstance(child, CustomTextItem):
                self.main_plot.getViewBox().removeItem(child)

        self.point_labeling_changed_signal.emit(annotations.get_number_labeled(self.main_plot_name))

    def update_labels(self):
        """
        Updates the labels of the annotated points
        Used if plot is changed
        """
        self._remove_all_labels()

        if self.showLabels == True:
            for labeled_point in annotations.get_labeled_points(self.main_plot_name):
                cti = CustomTextItem(index=labeled_point.index, text=labeled_point.label, point_x=labeled_point.x,
                                     point_y=labeled_point.val, color='ff0000', border='k', anchor=(0.5, 1.1), angle=0,
                                     fill='w')
                self.main_plot.getViewBox().addItem(cti)
                cti.setPos(int(labeled_point.x), labeled_point.val)

        self.point_labeling_changed_signal.emit(annotations.get_number_labeled(self.main_plot_name))

    @QtCore.pyqtSlot()
    def label_selection(self, lbl: str):
        """
        Labels the selected points with the selected label

        Parameters
        ----------
        lbl
        """
        annotations.add_labels_current_selection(self.main_plot_name, lbl)
        self.update_labels()
        self.deselect_all()

    @QtCore.pyqtSlot()
    def delabel_selection(self):
        """
        Removes the label from the selected points
        """
        annotations.delabel_selected(self.main_plot_name)
        self.update_labels()

    def _clicked_point_slot(self, plot: pg.PlotItem, points: list):
        """
        Slots that registers clicked plot points

        Parameters
        ----------
        plot
        points
        """
        indexes = []
        for point in points:
            point_position = point.pos()
            x, y = point_position.x(), point_position.y()
            lx = np.argwhere(self._x_data == x)
            ly = np.argwhere(self._y_data == y)
            i = np.intersect1d(lx, ly)[0]
            e = {'id': i, 'x': x, 'y': y}
            indexes.append(e)

        spot_items = self.main_dot_plot_item.points()
        for single_index in list({v['id']: v for v in indexes}.values()):
            point = annotations.SelectedPoint(single_index['y'], single_index['x'], single_index['id'])

            if annotations.is_point_selected(point):
                annotations.remove_selected_point(point)
                self._deselect(spot_items[single_index['id']])
            else:
                annotations.add_selection_point(point)
                self._select(spot_items[single_index['id']])

    def deselect_all(self):
        """
        Deselects all currently selected points
        """
        spot_items = self.main_dot_plot_item.points()
        current_selected_points = annotations.get_all_selection_points()

        for single_point in current_selected_points:
            single_spot_item = spot_items[single_point.index]
            self._deselect(single_spot_item)

        annotations.clear_selections()
        self.point_selection_changed_signal.emit(annotations.get_number_selected())

    def invert_selection(self):
        """
        Inverts all currently selected points
        """
        all_points = self.main_dot_plot_item.points()  # all points
        selected_points = annotations.get_all_selection_points()

        for single_point in all_points:
            lx = np.argwhere(self._x_data == single_point._data[0])
            ly = np.argwhere(self._y_data == single_point._data[1])
            lid = np.intersect1d(lx, ly)[0]
            point = annotations.SelectedPoint(single_point._data[1], single_point._data[0], int(lid))

            is_point_removed = False
            for selected_point in selected_points:
                if selected_point == point:
                    self._deselect(single_point)
                    annotations.remove_selected_point(selected_point)
                    is_point_removed = True
                    continue

            if not is_point_removed:
                self._select(single_point)
                annotations.add_selection_point(point)

    def _mouse_moved_event(self, evt):
        """
        Event to update the mouse location in the GUI

        Parameters
        ----------
        evt
        """
        position = evt[0]

        vb = self.main_plot.vb
        if self.main_plot.sceneBoundingRect().contains(position):
            mouse_point = vb.mapSceneToView(position)
            index = int(mouse_point.x())
            if index > 0:
                self.mouse_moved_signal.emit(mouse_point.x(), mouse_point.y())
            if self._v_line is not None:
                self._v_line.setPos(mouse_point.x())

            if self._h_line is not None:
                self._h_line.setPos(mouse_point.y())

    def export(self, type, path=None):
        """
        Exports the plot based on the selected format

        Parameters
        ----------
        type
        path
        """
        if type == "png":
            exporter = pg.exporters.ImageExporter(self.main_plot)
            exporter.parameters()['width'] = 1920
            exporter.export(path)

        elif type == "svg":
            exporter = pg.exporters.SVGExporter(self.main_plot)
            exporter.export(path)

        elif type == "mpl":
            from ndas.misc.colors import Color
            p_item = self.main_plot

            p_curve_list = []
            for i in p_item.curves:
                i.opts['symbolPen'] = Color.BLUE.value
                i.opts['symbolBrush'] = Color.BLUE.value
                i.opts['symbolSize'] = 1
                i.opts['fillLevel'] = None
                i.opts['fillBrush'] = None
                p_curve_list.append(i)

            p_item.curves = p_curve_list
            exporter = pg.exporters.MatplotlibExporter(p_item)
            exporter.export()


class MultiSelectViewBox(pg.ViewBox):
    """
    Viewbox for multiselection of points
    """
    multipoint_selection_changed_signal = QtCore.pyqtSignal(tuple)

    def mouseDragEvent(self, ev):
        """
        Overwrites the mouseDragEvent from pg.ViewBox with custom rectangle draw event
        for multipoint selection

        Parameters
        ----------
        ev
        """
        modifiers = QApplication.keyboardModifiers()

        if ev.button() == QtCore.Qt.RightButton and (modifiers == QtCore.Qt.ControlModifier):
            ev.accept()
            rect = self.parentItem().items[0]
            rect.hide()
            updated_rect = QtCore.QRectF(self.mapToView(self.mapFromParent(ev.buttonDownPos())), self.mapToView(self.mapFromParent(ev.pos())))

            if ev.isFinish():
                rect_coordinates = updated_rect.getCoords()

                bottom_left = []
                top_right = []
                if rect_coordinates[0] < rect_coordinates[2]:
                    bottom_left.append(rect_coordinates[0])
                    top_right.append(rect_coordinates[2])
                else:
                    bottom_left.append(rect_coordinates[2])
                    top_right.append(rect_coordinates[0])
                if rect_coordinates[1] < rect_coordinates[3]:
                    bottom_left.append(rect_coordinates[1])
                    top_right.append(rect_coordinates[3])
                else:
                    bottom_left.append(rect_coordinates[3])
                    top_right.append(rect_coordinates[1])

                self.signal_selection_change((bottom_left, top_right))
            else:
                rect.setPos(updated_rect.topLeft())
                rect.setTransform(QTransform.fromScale(updated_rect.width(), updated_rect.height()))
                rect.show()
        else:
            pg.ViewBox.mouseDragEvent(self, ev)

    def signal_selection_change(self, selection_list):
        """
        Signals a change in the selection, after the rectangular has been drawn.

        Parameters
        ----------
        selection_list
        """
        self.multipoint_selection_changed_signal.emit(selection_list)


class CustomTextItem(pg.TextItem):
    """
    Textitem, that is used to display annotations in plot.
    """

    def __init__(self, index, point_x, point_y, text, angle=0, rotateAxis=None, anchor=(0, 0), *args, **kwargs):
        """
        Creates the label

        Parameters
        ----------
        index
        point_x
        point_y
        text
        angle
        rotateAxis
        anchor
        args
        kwargs
        """
        super().__init__(text=text, angle=angle, rotateAxis=rotateAxis, anchor=anchor, *args, **kwargs)
        self.point_index = index
        self.text = text
        self.point_x = point_x
        self.point_y = point_y

    def __eq__(self, other):
        """
        Compare function for different instances of CustomTextItem

        Parameters
        ----------
        other
        """
        if isinstance(other, int):
            return self.point_index == other
        elif isinstance(other, CustomTextItem):
            return self.point_index == other.point_index and self.text == other.text
