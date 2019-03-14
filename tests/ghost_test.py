from ghost import Ghost
from time import sleep
from PyQt4.QtGui import *
ghost = Ghost()

with ghost.start() as session:
    page, extra_resources = session.open("http://reddit.com")
    session.show()
    sleep(5)
    p = QPixmap.grabWindow(session.webview.winId())
    p.save('test.jpg', 'jpg')

