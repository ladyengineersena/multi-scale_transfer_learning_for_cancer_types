#datetime modülü 2.5
from datetime import datetime
from datetime import timedelta

suan = datetime.now()
tdelta = timedelta(days=8,hours=5,seconds=61)
#1. bulunduğum zamandan itibaren bu kadar süre geçerse n'olur
print(suan + tdelta)
#2.se  bulunduğum zamandan bu kadar önce neydi 
print(suan - tdelta)

print(suan.month)
print(suan.date())
print(suan.day)
print(suan.year)
print(suan.second)
print(suan.time())