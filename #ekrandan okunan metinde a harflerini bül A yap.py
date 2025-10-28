#ekrandan okunan metinde a harflerini büyük yapan bir program
metin = input("bir metin giriniz:")
metin2 = ""

for harf in metin:
    if harf == "a":
        metin2 += "A"
    else:
        metin2 += harf
print(metin2)    
