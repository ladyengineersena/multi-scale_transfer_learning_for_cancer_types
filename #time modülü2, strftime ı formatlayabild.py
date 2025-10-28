#time modülü2, strftime ı formatlayabildiğimiz içeriktir.
import time
zaman = time.strftime("%d:%m:%Y: %H:%M:%S %p")
print(zaman)
#time.sleep fonksiyonu aradaki süreyi belirler.
print("Program başlatıldı")
time.sleep(3)
print("Program sonlandırıldı.")
