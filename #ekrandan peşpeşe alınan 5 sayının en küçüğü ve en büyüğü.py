#ekrandan peşpeşe alınan 5 sayının en küçüğünü ve en büyüğünü ekrana yazdıran bir program yazınız
liste = []
for i in range(5):
    sayı = int(input("bir sayı giriniz:"))
    liste.append(sayı)
print(f"en büyük sayı: {max(liste)}")
print(f"en küçük sayı: {min(liste)}")
