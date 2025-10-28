#ekrandan alınan sayının bir sayının karesi olup olmadığını kontrol eden program
sayı = int(input("bir sayı giriniz:"))
karekök = sayı ** 0.5
if karekök == int(karekök):
    print("tam kare")
else:
    print("tam kare değil")



