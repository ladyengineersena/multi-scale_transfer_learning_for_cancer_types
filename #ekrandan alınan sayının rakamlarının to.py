#ekrandan alınan sayının rakamlarının toplamını yazınız(for döngüsüyle stringlerde çalışabilim)
sayı= int(input("bir sayı giriniz:"))
str_sayı = str(sayı)
toplam = 0
for rakam in str_sayı:
    toplam += int(rakam)
print(toplam)
