#random, uniform fonksiyonları
#randint ve randrange fonksiyonları
#choice,shuffle ve sample fonksiyonları
#bir modülün tüm fonksiyonlarını bilmek mümkün değil
import random
for i in range(10):
    print(random.random())

for a in range(10):
    print(random.uniform(10,30))
#randint fonksiyonu ilk ve son sınırıda dahil edip gösterir.
for b in range(10):
    print(random.randint(1,5))
#randrange fonksiyonu üst sınırı dahil etmez.
for c in range(10):
    print(random.randrange(1,10,2))


import random
liste = ["mavi","yesil","sarı","mor"]
print(random.choice(liste))
#shuffle fonksiyonu yer değiştirir.
random.shuffle(liste)
print(liste)

print(random.sample(liste,3))