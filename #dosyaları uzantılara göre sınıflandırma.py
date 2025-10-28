#dosyaları uzantılara göre sınıflandırma
import os 
def düzenle():
    klasör = input("düzenlenecek klasör: ")
    dosyalar = [] #dosyalar depolanacak
    uzantılar = [] #uzantılar depolanacak
    def list_dir():
        for dosya in os.listdir(klasör):
            if os.path.isdir(os.path.join(klasör,dosya)): #dosyamız klasör mü?
               continue
            if dosya.startswith("."): #dosyamız bir gizli dosya mı?
               continue
            else:
                dosyalar.append(dosya)
    list_dir()            
    #uzantıları alma
    for dosya in dosyalar:
        uzantı = dosya.split(".")[-1]
        if uzantı in uzantılar: #uzantılar daha önce eklendi mi?
            continue
        else:
            uzantılar.append(uzantı)
    for uzantı in uzantılar: #klasörler oluşturuluyor
        if os.path.isdir(os.path.join(klasör,uzantı)):
            continue
        else:
            os.mkdir(os.path.join(klasör,uzantı))
    for dosya in dosyalar:
        uzantı = dosya.split(".")[-1]
        os.rename(os.path.join(klasör,dosya),os.path.join(klasör,uzantı,dosya))
if __name__ == "__main__":
    düzenle()

