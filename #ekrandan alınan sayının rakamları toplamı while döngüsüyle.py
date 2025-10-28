#ekrandan alınan sayının rakamları toplamı while döngüsüyle
sayı = int(input("bir sayı giriniz"))
toplam = 0
geçici_sayı = sayı
while geçici_sayı != 0:
    basamak = geçici_sayı % 10
    toplam += basamak
    geçici_sayı //= 10

print(toplam)