"""Layout and theme helpers shared by the Bootstrapper and Proofreading
widgets."""

from napari.utils.theme import get_theme, template
from qtpy.QtCore import QTimer
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpacerItem,
    QVBoxLayout,
)
from superqt import QCollapsible

# The section margin plus the 1 px border that napari's QSS draws around a
# QGroupBox. Rows outside a section use it so their edges match.
SECTION_INSET = 9

# Rows inside a group sit 6 px apart. A new group gets 12 px above it, and
# the content under a disclosure header is indented by 12 px.
GROUP_SPACING = 12
GROUP_INDENT = 12

PLUGIN_QSS = """
QGroupBox[section="true"] {
  /* The sections have no QGroupBox title, so they need no title margin.
     napari gives every QWidget 1 px of padding, which a QGroupBox adds
     inside its border, so it is 0 here to keep SECTION_INSET exact. */
  margin-top: 0px;
  padding: 0px;
  /* napari's foreground border is 1.1 to 1.2:1 against the background.
     This one is about 1.6:1 in the light theme and 2:1 in the dark. */
  border-color: {{ lighten(primary, 5) }};
}

QLabel[sectionTitle="true"] {
  font-weight: bold;
  font-size: {{ increase(font_size, 1) }};
}

QPushButton[compact="true"] {
  padding: 2px 8px;
  font-size: {{ font_size }};
}

QPushButton[primary="true"]:enabled {
  background-color: {{ current }};
}

QPushButton[primary="true"]:enabled:hover {
  background-color: {{ lighten(current, 10) }};
}

QPushButton[primary="true"]:enabled:pressed {
  background-color: {{ darken(current, 10) }};
}

QToolButton[disclosure="true"],
QCollapsible > QPushButton {
  background-color: transparent;
  border: none;
  padding: 2px 0px;
}

QCollapsible {
  /* Without napari's 1 px padding the dialog groups sit GROUP_SPACING
     apart. */
  padding: 0px;
}

QCollapsible > QPushButton {
  font-weight: bold;
}

QToolButton[disclosure="true"]:hover,
QCollapsible > QPushButton:hover {
  background-color: {{ foreground }};
}

QLineEdit:read-only {
  color: {{ opacity(text, 150) }};
}
"""


def current_theme(viewer):
    """Returns the napari theme of the viewer, or the dark theme if napari
    does not know the viewer's theme."""
    try:
        return get_theme(viewer.theme)
    except ValueError:
        return get_theme("dark")


def plugin_stylesheet(theme):
    return template(PLUGIN_QSS, **theme.to_rgb_dict())


def apply_plugin_theme(widget, theme, align_labels):
    """
    Sets the plugin stylesheet for `theme` on `widget`. napari restyles its
    window in its own theme handler, so a visible widget calls
    `align_labels` after that handler runs.
    """
    widget.setStyleSheet(plugin_stylesheet(theme))
    if widget.isVisible():
        QTimer.singleShot(0, align_labels)


def format_label_grid(grid, inset=0):
    """Spaces a grid of labels in column 0 and fields in column 1, and
    gives the spare width to the fields."""
    grid.setContentsMargins(inset, 0, inset, 0)
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(6)
    grid.setColumnStretch(0, 0)
    grid.setColumnStretch(1, 1)


def make_action_grid(rows):
    """
    Returns a grid of three equal columns. `rows` lists the widgets of each
    row with the number of columns they span, for example
    [[(start, 2), (stop, 1)]].
    """
    grid = QGridLayout()
    grid.setSpacing(6)
    for column in range(3):
        grid.setColumnStretch(column, 1)
    for row, cells in enumerate(rows):
        column = 0
        for widget, span in cells:
            grid.addWidget(widget, row, column, 1, span)
            column += span
    # Qt leaves out the gap between two columns that no row splits, which
    # widens the thirds of a grid whose buttons span two columns. A row of
    # empty spacers splits every column and takes no height.
    for column in range(3):
        grid.addItem(QSpacerItem(0, 0), len(rows), column)
    return grid


def make_section(title, body, tooltip="", button=None):
    """
    Returns a framed section with a bold title row above the `body` layout.
    The optional `button` sits at the right end of the title row.
    """
    box = QGroupBox()
    box.setProperty("section", True)
    box.setAccessibleName(title)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(8, 6, 8, 8)
    layout.setSpacing(6)

    title_label = QLabel(title)
    title_label.setProperty("sectionTitle", True)
    title_label.setToolTip(tooltip)
    title_row = QHBoxLayout()
    title_row.addWidget(title_label)
    title_row.addStretch(1)
    if button is not None:
        button.setProperty("compact", True)
        title_row.addWidget(button)
    layout.addLayout(title_row)

    body.setContentsMargins(0, 0, 0, 0)
    if isinstance(body, QGridLayout):
        format_label_grid(body)
    layout.addLayout(body)
    return box


def align_label_columns(labels, grids):
    """
    Gives column 0 of every grid the width of the widest label. napari's
    stylesheet sets the label fonts only once a widget sits in its dock, so
    the widgets call this from showEvent and again after a theme change.
    The Proofreading widget also calls it when its operation group changes.
    """
    for label in labels:
        label.ensurePolished()
    width = max(label.sizeHint().width() for label in labels)
    for grid in grids:
        grid.setColumnMinimumWidth(0, width)


def make_collapsible(title, theme):
    collapsible = QCollapsible(title)
    collapsible.layout().setContentsMargins(0, 0, 0, 0)
    collapsible.layout().setSpacing(0)
    collapsible.content().layout().setContentsMargins(GROUP_INDENT, 0, 0, 0)
    # superqt paints string icons in the toggle's WindowText colour when
    # they are set, so the colour must come from the napari theme first.
    toggle = collapsible.toggleButton()
    palette = toggle.palette()
    palette.setColor(QPalette.WindowText, QColor(theme.text.as_hex()))
    toggle.setPalette(palette)
    collapsible.setCollapsedIcon("▶")
    collapsible.setExpandedIcon("▼")
    return collapsible


def style_loss_plot(plot, theme, pen):
    plot.setBackground(theme.foreground.as_hex())
    for name in ("left", "bottom"):
        axis = plot.getAxis(name)
        axis.setPen(theme.icon.as_hex())
        axis.setTextPen(theme.icon.as_hex())
    plot.getAxis("left").enableAutoSIPrefix(False)
    label_style = {"color": theme.text.as_hex(), "font-size": "12px"}
    plot.setLabel("left", "Loss", **label_style)
    plot.setLabel("bottom", "Iterations", **label_style)
    for item in plot.listDataItems():
        item.setPen(pen)
