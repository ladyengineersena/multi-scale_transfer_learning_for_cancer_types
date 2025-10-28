#dosyaları zamana göre sınıflandırma
import os 
from datetime import datetime 
def düzenle():
    klasör = input("düzenlenecek klasör: ")
    dosyalar = [] #dosyalar depolanacak
    tarihler = [] #tarihler depolanacak
    def list_dir():
        for dosya in os.listdir(klasör):
            if os.path.isdir(os.path.join(klasör,dosya)): #dosyamız klasör mü?
               continue
            if dosya.startswith("."): #dosyamız bir gizli dosya mı?
               continue
            else:
                dosyalar.append(dosya)
    list_dir()            
    #tarihleri alma 
    for dosya in dosyalar:
        tarih_damgası = os.stat(os.path.join(klasör,dosya)).st_birthtime
        tarih = datetime.fromtimestamp(tarih_damgası).strftime("%d--%m--%Y")
        if tarih in tarihler:
            continue
        else:
            tarihler.append(tarih)
    for tarih in tarihler: #klasörler oluşturuluyor
        if os.path.isdir(os.path.join(klasör,tarih)):
            continue
        else:
            os.mkdir(os.path.join(klasör,tarih))
    for dosya in dosyalar:
        tarih_damgası = os.stat(os.path.join(klasör,dosya)).st_birthtime
        tarih = datetime.fromtimestamp(tarih_damgası).strftime("%d--%m--%Y")
        
        os.rename(os.path.join(klasör,dosya),os.path.join(klasör,tarih,dosya))
if __name__ == "__main__":
    düzenle()

