#fonksiyonlar aynı işlevi daha kısa sürede yapmamızı sağlar
def bilgi_ver():
    print("işlem başarılı!")

bilgi_ver()
bilgi_ver()
bilgi_ver()

def selamla(isim):
    print("Merhaba" +  isim)

selamla("Ali")
#iki parametreli
def carp(x,y):
    print(f"x*y = {x*y}")

carp(3,6)

def ortalama_hesapla(liste):
    toplam = sum(liste)
    adet = len(liste)
    ortalama = toplam  / adet
    print(f"Girilen sayıların ortalaması: {ortalama}")
ortalama_hesapla([1,2,3,4,5,6,7])    

def büyükharfe_cevir(metin):
    metin = metin.upper()
    print(metin)
büyükharfe_cevir("Banane")

def selamla(mesaj,isim):
    print(f"{mesaj} {isim}")
selamla("merhaba","Sena")   

#varsayılan parametrede 2.yi yazarsan anonim olmaz,varsayılan olur.
def selamla(mesaj,isim = "Anonim"):
    print(f"{mesaj} {isim}")
selamla("merhaba","Kübra")

def indirim_yap(fiyat,yüzde = 20):
    indirim_miktarı = fiyat * (yüzde/100)
    indirimli_fiyat = fiyat - indirim_miktarı
    print(f"indirimli_tutar: {indirimli_fiyat}")
indirim_yap(120,30)