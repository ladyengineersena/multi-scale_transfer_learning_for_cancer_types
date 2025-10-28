#3 basamaklı sayıların kaç tanesi rakamlarının küplerinin toplamına eşittir?
liste = []
for sayı in range(100,1000):
    toplam = 0
    gecici_sayı = sayı
    while gecici_sayı!= 0:
        #birler basamağina baktık
        basamak = gecici_sayı % 10
        #toplamı kübü kadar artırdık
        toplam += basamak ** 3
        gecici_sayı //= 10
    if toplam == sayı:
        liste.append(sayı)
print(liste)
