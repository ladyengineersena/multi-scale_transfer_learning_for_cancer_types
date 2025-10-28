#modüller = paket,içine fonksiyon yazmak için
import math
sonuc = math.factorial(4)
print(sonuc)
sonuc2 = math.sqrt(81)
print(sonuc2)
sonuc3 = math.fabs(64)

print(int(sonuc3))
#import ettiğini sonucun yanına yazıyosun
from math import factorial
sonuc4 = factorial(5)
print(sonuc4)

from math import factorial, sqrt
sonuc5 = factorial(4),sqrt(5)
print(sonuc5)

from math import *
sonuc6 = sqrt(8)
sonuc7 = fabs(7)
print(sonuc6)
print(sonuc7)