#pisagore göre a^2 + b^2 = c^2 , a+b+c = 1000, find product (carpim) of abc.
done = False
for a in range(1,1000):
 for b in range (1,1000-a):
     c = 1000 - a -b
     if c*c == a*a + b*b:
         print(a*b*c)
         done = True
         break
#dıştaki fordanda çıkalım
 if done:
        break
