#date time modülü 2
from datetime import datetime


suan = datetime.now()
print(suan)
print(suan.ctime())
print(datetime.ctime(suan))
print(suan.date())
print(suan.time())
print(suan.date().month)

suan2 = datetime.now()
gecmis_an = datetime(2018,4,12,3,15,20,39)
print(suan2 - gecmis_an)

bugun = date.today()
suan3 = datetime.now()
print(bugun.strftime("%d:%m:%Y  %A"))
print(suan.strftime("%d:%m:%Y %A"))

print(datetime.strftime(bugun,"%d:%m:%Y"))
print(suan.strftime("%d:%m:%Y"))

