#ilk 8  printten gelen,ikinci 8 print sonuctan gelen.
def topla(x,y):
    print(x + y)
    return x + y
sonuc = topla(3,5)
print(sonuc)

def ortalama_hesapla(a,b):
    return (a + b) / 2
print(ortalama_hesapla(3,7))

print(type(ortalama_hesapla))
print(type(ortalama_hesapla(4,6)))
c = ortalama_hesapla(2,8)
d = ortalama_hesapla(6,6)
print (c+d)

def büyük_harfe_cevir(metin):
   return metin.upper()
q = büyük_harfe_cevir("Bizenebundan")
print(q)

