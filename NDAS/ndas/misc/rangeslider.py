from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class RangeSlider(QSlider):
    """
    The data slicing tool based on the QSlider
    """
    valueChanged = pyqtSignal(int, int)

    def __init__(self, parent=None):
        """
        Creates the slider and defines the tick interval

        Parameters
        ----------
        parent
        """
        super().__init__(parent)

        self.first_position = 0
        self.second_position = 100

        self.opt = QStyleOptionSlider()
        self.opt.minimum = 0
        self.opt.maximum = 100

        self.setTickPosition(QSlider.TicksAbove)
        self.setTickInterval(10)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed, QSizePolicy.Slider)
        )

    def setRangeLimit(self, minimum: int, maximum: int):
        """
        Sets the maximum and minimum values (range)

        Parameters
        ----------
        minimum
        maximum
        """
        self.opt.minimum = minimum
        self.opt.maximum = maximum
        self.setRange(minimum, maximum)
        self.setTickInterval(int(maximum / 30))

    def setRange(self, start: int, end: int):
        """
        Sets the current range

        Parameters
        ----------
        start
        end
        """
        self.first_position = start
        self.second_position = end
        self.update()

    def getRange(self):
        """
        Returns the current range

        """
        return (self.first_position, self.second_position)

    def setTickPosition(self, position: QSlider.TickPosition):
        """
        Sets the tick position

        Parameters
        ----------
        position
        """
        self.opt.tickPosition = position

    def setTickInterval(self, ti: int):
        """
        Sets the tick interval

        Parameters
        ----------
        ti
        """
        self.opt.tickInterval = ti

    def paintEvent(self, event: QPaintEvent):
        """
        Draws this custom slicer

        Parameters
        ----------
        event
        """

        painter = QPainter(self)

        self.opt.initFrom(self)
        self.opt.rect = self.rect()
        self.opt.sliderPosition = 0
        self.opt.subControls = QStyle.SC_SliderGroove | QStyle.SC_SliderTickmarks

        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

        color = self.palette().color(QPalette.Highlight)
        color.setAlpha(50)
        painter.setBrush(QBrush(color))

        painter.setPen(Qt.NoPen)

        self.opt.sliderPosition = self.first_position
        x_left_handle = (
            self.style()
                .subControlRect(QStyle.CC_Slider, self.opt, QStyle.SC_SliderHandle)
                .right()
        )

        self.opt.sliderPosition = self.second_position
        x_right_handle = (
            self.style()
                .subControlRect(QStyle.CC_Slider, self.opt, QStyle.SC_SliderHandle)
                .left()
        )

        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, self.opt, QStyle.SC_SliderGroove
        )

        selection = QRect(
            x_left_handle,
            groove_rect.y(),
            x_right_handle - x_left_handle,
            groove_rect.x(),
        ).adjusted(-1, 10, 1, 5)

        painter.drawRect(selection)

        self.opt.subControls = QStyle.SC_SliderHandle
        self.opt.sliderPosition = self.first_position
        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

        self.opt.sliderPosition = self.second_position
        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

    def mousePressEvent(self, event: QMouseEvent):
        """
        Event to access clicks via mouse

        Parameters
        ----------
        event
        """

        self.opt.sliderPosition = self.first_position
        self._first_sc = self.style().hitTestComplexControl(
            QStyle.CC_Slider, self.opt, event.pos(), self
        )

        self.opt.sliderPosition = self.second_position
        self._second_sc = self.style().hitTestComplexControl(
            QStyle.CC_Slider, self.opt, event.pos(), self
        )

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Event to access movements via mouse

        Parameters
        ----------
        event
        """

        distance = self.opt.maximum - self.opt.minimum

        pos = self.style().sliderValueFromPosition(
            0, distance, event.pos().x(), self.rect().width()
        )

        if self._first_sc == QStyle.SC_SliderHandle:
            if pos <= self.second_position:
                self.first_position = pos
                self.update()
                self.valueChanged.emit(self.first_position, self.second_position)
                return

        if self._second_sc == QStyle.SC_SliderHandle:
            if pos >= self.first_position:
                self.second_position = pos
                self.update()
                self.valueChanged.emit(self.first_position, self.second_position)

    def sizeHint(self):
        """
        Sets the custom ticks
        """
        SliderLength = 84
        TickSpace = 5

        w = SliderLength
        h = self.style().pixelMetric(QStyle.PM_SliderThickness, self.opt, self)

        if (
                self.opt.tickPosition & QSlider.TicksAbove
                or self.opt.tickPosition & QSlider.TicksBelow
        ):
            h += TickSpace

        return (
            self.style()
                .sizeFromContents(QStyle.CT_Slider, self.opt, QSize(w, h), self)
                .expandedTo(QApplication.globalStrut())
        )
