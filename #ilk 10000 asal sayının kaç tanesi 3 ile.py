#ilk 10000 asal sayının kaç tanesi 3 ile başlar 7 ile  biter?
prime_list = list()
prime_list.append(2)
#3 ten itibaren bütün sayıları asal olup olmadığını kontrol edecek
sayı = 3
#ben çık demediğim sürece döngüyü devam ettir
while True:
    prime = True
    for i in range(2,sayı):
        if sayı %i == 0:
            prime = False
           #break for döngüsünden çıkarıcak. 
            break
    if prime:
        prime_list.append(sayı)
        if len(prime_list) == 10000:
            break
    sayı += 1
liste2 = []
for prime in prime_list:
    strprime = str(prime)
    if strprime.startswith("3") and strprime.endswith("7"):
      liste2.append(prime)
print(liste2)
print(len(liste2))