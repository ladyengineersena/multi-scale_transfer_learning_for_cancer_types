#time modülü
import time
baslangıc = time.time()
liste = []
for i in range(100000):
    liste.append(i) 
bitis = time.time()
print(bitis - baslangıc)

#tarih saniye olarak algılanıyor
zaman = time.ctime(100000000)
print(zaman)
print(type(zaman))

#local time time tuple olarak veriyor.
zaman2 = time.localtime()
print(zaman2)

zaman3 = time.localtime()
zaman4 = time.asctime(zaman3)
print(zaman4)