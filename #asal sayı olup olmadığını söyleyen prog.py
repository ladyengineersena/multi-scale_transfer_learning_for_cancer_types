#asal sayı olup olmadığını söyleyen program
sayı = int(input("bir sayı giriniz:"))
#bütün sayılara asal muamelesi yaptık asal değişkeniyle
prime = True
#hersayı 1 e ve kendisine bölündüğü için asalları bulamayız. o yüzden 2 den baslayıp sayı+1 yapmadık.
#2 den sayıya kadar bölünüyosa asal değil.
for i in range(2,sayı):
 if sayı %i == 0:
     prime = False
     break
if prime == True:
    print(f"{sayı} sayısı asaldır")
else:
    print(f"{sayı} sayısı asal değildir")

