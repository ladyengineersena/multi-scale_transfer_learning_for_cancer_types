import random
#kütüphane oluşturdum
zarlar = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

for i in range(100):
# 1 ve 6 arasında rastgele değer seçip onun karşılığını 1 artırıcak.
    zar = random.randint(1,6)
    zarlar[zar] += 1

for zar in zarlar:
    print(f"{zar} gelme olasılığı: {zarlar[zar] / 100}")
 
altı_altı = 0
deneme_sayısı = 0 
while True:
    deneme_sayısı +=1
    zar1 =random.randint(1,6)
    zar2 =random.randint(1,6)
    if zar1 == 6 and zar2 == 6:
        altı_altı +=1
    if altı_altı == 10:
       print(f"10 kere 6-6 gelmesi için zarlar {deneme_sayısı} kadar atıldı.")     
       break
