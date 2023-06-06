
#.ui uzantılı dosyayı aşağıdaki kodla .py uzantısına dönüştürdüm
from PyQt5 import uic

with open('arayuzui.py', 'w', encoding="utf-8") as fout:
   uic.compileUi('arayuz.ui', fout)
