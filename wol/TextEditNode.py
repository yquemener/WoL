from PyQt5.QtCore import QRegExp, Qt, QRect, QObject
from PyQt5.QtGui import QOpenGLTexture, QImage, QSyntaxHighlighter, QColor, QTextCharFormat, QFont
from PyQt5.QtWidgets import QTextEdit, QApplication

from wol import utils
from wol.Constants import Events
from wol.GeomNodes import CardNode
import re


def test():
    print("Yo")


class TextEditNode(CardNode):
    def __init__(self, parent, name="GuiNode", text="", autosize=False):
        CardNode.__init__(self, name=name, parent=parent)
        self.widget = QTextEdit()
        self.max_geometry = QRect(0, 0, 512, 512)
        self.widget.setGeometry(self.max_geometry)
        self.focusable = True
        self.widget.setText(text)
        self.text = text
        self.autosize = autosize
        self.min_size = (-1, -1)
        if self.autosize:
            self.do_autosize()
        self.needs_refresh = True
        self.highlight = PythonHighlighter(self.widget.document())
        self.widget.setTextColor(QColor(255, 255, 255))
        self.widget.setStyleSheet("QWidget{color: white; background-color: black;}");
        qfm = self.widget.fontMetrics()
        self.widget.setTabStopDistance(qfm.horizontalAdvance(' ') * 4)
        self.focused = False

    def do_autosize(self):
        qfm = self.widget.fontMetrics()
        rect = qfm.boundingRect(self.max_geometry, Qt.TextWordWrap, self.text, 4)
        w = rect.width() + 30
        h = rect.height() + 30
        if self.min_size[0] > 0:
            w = max(self.min_size[0], w)
        if self.min_size[1] > 0:
            h = max(self.min_size[1], h)
        self.widget.setGeometry(0, 0, w, h)
        self.hscale = h / 512.0
        self.wscale = w / 512.0
        self.vertices = utils.generate_square_vertices_fan()
        for v in self.vertices:
            v[1] *= self.hscale
            v[0] *= self.wscale
        self.refresh_vertices()

    def update(self, dt):
        if self.focused:
            self.needs_refresh = True
        if self.needs_refresh:
            if self.autosize:
                self.do_autosize()
            self.texture = QOpenGLTexture(QImage(self.widget.grab()))
            self.needs_refresh = False

    def on_unfocus(self):
        self.focused = False

    def keyPressEvent(self, evt):
        if evt.key() == Qt.Key_Return:
            cursi = self.widget.textCursor().position()
            lstart = self.text.rfind("\n", 0, cursi)
            whitespaces_re = re.compile("^([ \t]*)")
            matches = whitespaces_re.match(self.text[lstart:cursi].lstrip("\n"))
            if not matches is None:
                self.widget.keyPressEvent(evt)
                self.widget.insertPlainText(matches.group())
            else:
                self.widget.keyPressEvent(evt)
        else:
            self.widget.keyPressEvent(evt)
        self.text = self.widget.toPlainText()
        self.on_event(Events.TextChanged)

    def inputMethodEvent(self, evt):
        return self.widget.inputMethodEvent(evt)

    def set_text(self, t):
        if t == self.text:
            return
        self.text = t
        self.widget.setText(t)
        if self.autosize:
            self.do_autosize()
        self.needs_refresh = True
        self.on_event(Events.TextChanged)


class ErrorWindow(TextEditNode):
    def __init__(self, parent, text):
        super().__init__(parent=parent, text=text, autosize=True)
        self.highlight.setDocument(None)
        self.widget.setStyleSheet("""
                color: rgba(255,255,255,255);
                background-color: rgba(0,0,0,100);
                border: 2px solid rgba(255,0,0,255);;
            """)


def formatter(color, style=''):
    """Return a QTextCharFormat with the given attributes.
    """
    _color = QColor(255, 255, 255)
    _color.setNamedColor(color)

    _format = QTextCharFormat()
    _format.setForeground(_color)
    if 'bold' in style:
        _format.setFontWeight(QFont.Bold)
    if 'italic' in style:
        _format.setFontItalic(True)

    return _format


# Syntax styles that can be shared by all languages
STYLES = {
    'comment': formatter('green', 'italic'),
    'string': formatter('magenta'),
    'string2': formatter('magenta'),
    'keyword': formatter('blue'),
    'operator': formatter('red'),
    'brace': formatter('lightGray'),
    'defclass': formatter('white', 'bold'),
    'self': formatter('white', 'italic'),
    'numbers': formatter('yellow'),
}


class PythonHighlighter (QSyntaxHighlighter):
    """Syntax highlighter for the Python language.
    """
    # Python keywords
    keywords = [
        'and', 'assert', 'break', 'class', 'continue', 'def',
        'del', 'elif', 'else', 'except', 'exec', 'finally',
        'for', 'from', 'global', 'if', 'import', 'in',
        'is', 'lambda', 'not', 'or', 'pass', 'print',
        'raise', 'return', 'try', 'while', 'yield',
        'None', 'True', 'False',
    ]

    # Python operators
    operators = [
        '=',
        # Comparison
        '==', '!=', '<', '<=', '>', '>=',
        # Arithmetic
        '\+', '-', '\*', '/', '//', '\%', '\*\*',
        # In-place
        '\+=', '-=', '\*=', '/=', '\%=',
        # Bitwise
        '\^', '\|', '\&', '\~', '>>', '<<',
    ]

    # Python braces
    braces = [
        '\{', '\}', '\(', '\)', '\[', '\]',
    ]

    def __init__(self, document):
        QSyntaxHighlighter.__init__(self, document)

        # Multi-line strings (expression, flag, style)
        # FIXME: The triple-quotes in these two lines will mess up the
        # syntax highlighting from this point onward
        self.tri_single = (QRegExp("'''"), 1, STYLES['string2'])
        self.tri_double = (QRegExp('"""'), 2, STYLES['string2'])

        rules = []

        # Keyword, operator, and brace rules
        rules += [(r'\b%s\b' % w, 0, STYLES['keyword'])
            for w in PythonHighlighter.keywords]
        rules += [(r'%s' % o, 0, STYLES['operator'])
            for o in PythonHighlighter.operators]
        rules += [(r'%s' % b, 0, STYLES['brace'])
            for b in PythonHighlighter.braces]

        # All other rules
        rules += [
            # 'self'
            (r'\bself\b', 0, STYLES['self']),

            # 'def' followed by an identifier
            (r'\bdef\b\s*(\w+)', 1, STYLES['defclass']),
            # 'class' followed by an identifier
            (r'\bclass\b\s*(\w+)', 1, STYLES['defclass']),

            # From '#' until a newline
            (r'#[^\n]*', 0, STYLES['comment']),

            # Numeric literals
            (r'\b[+-]?[0-9]+[lL]?\b', 0, STYLES['numbers']),
            (r'\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b', 0, STYLES['numbers']),
            (r'\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b', 0, STYLES['numbers']),

            # Double-quoted string, possibly containing escape sequences
            (r'"[^"\\]*(\\.[^"\\]*)*"', 0, STYLES['string']),
            # Single-quoted string, possibly containing escape sequences
            (r"'[^'\\]*(\\.[^'\\]*)*'", 0, STYLES['string']),

        ]

        # Build a QRegExp for each pattern
        self.rules = [(QRegExp(pat), index, fmt)
                      for (pat, index, fmt) in rules]

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text.
        """
        # Do other syntax formatting
        for expression, nth, format in self.rules:
            index = expression.indexIn(text, 0)

            while index >= 0:
                # We actually want the index of the nth match
                index = expression.pos(nth)
                length = len(expression.cap(nth))
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

        self.setCurrentBlockState(0)

        # Do multi-line strings
        in_multiline = self.match_multiline(text, *self.tri_single)
        if not in_multiline:
            in_multiline = self.match_multiline(text, *self.tri_double)

    def match_multiline(self, text, delimiter, in_state, style):
        """Do highlighting of multi-line strings. ``delimiter`` should be a
        ``QRegExp`` for triple-single-quotes or triple-double-quotes, and
        ``in_state`` should be a unique integer to represent the corresponding
        state changes when inside those strings. Returns True if we're still
        inside a multi-line string when this function is finished.
        """
        # If inside triple-single quotes, start at 0
        if self.previousBlockState() == in_state:
            start = 0
            add = 0
        # Otherwise, look for the delimiter on this line
        else:
            start = delimiter.indexIn(text)
            # Move past this match
            add = delimiter.matchedLength()

        # As long as there's a delimiter match on this line...
        while start >= 0:
            # Look for the ending delimiter
            end = delimiter.indexIn(text, start + add)
            # Ending delimiter on this line?
            if end >= add:
                length = end - start + add + delimiter.matchedLength()
                self.setCurrentBlockState(0)
            # No; multi-line string
            else:
                self.setCurrentBlockState(in_state)
                length = len(text) - start + add
            # Apply formatting
            self.setFormat(start, length, style)
            # Look for the next match
            start = delimiter.indexIn(text, start + length)

        # Return True if still inside a multi-line string, False otherwise
        if self.currentBlockState() == in_state:
            return True
        else:
            return False
