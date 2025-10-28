#date time modülü
from datetime import date
bugun = date.today()
print(bugun)
print(bugun.day)
print(bugun.month)
print(bugun.year)
print(bugun.weekday())
#weekday 0 dan başlarken isoweektey 1 den baslar.
print(bugun.isoweekday())

gecmis_tarih = date(2013,7,3)
print(gecmis_tarih.weekday())
gecen_zaman = bugun - gecmis_tarih
print(gecen_zaman)
print(type(gecen_zaman))

from datetime import datetime
suan = datetime.now()
print(suan)
print(type(suan))
print(suan.year)
print(suan.day)
print(suan.hour)
print(suan.minute)