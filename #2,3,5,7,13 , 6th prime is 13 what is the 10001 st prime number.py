#2,3,5,7,13 , 6th prime is 13 what is the 10001st prime number?
import math
prime_count = 0
def prime_check(x):
 is_prime = True
 if x == 2:
     return True
 else:
    for i in range(2,int(math.sqrt(x)+1)):
         if x %i == 0:
             is_prime = False
             break
    return is_prime 
i= 2
while True:
    if prime_check(i):
        prime_count += 1
    if prime_count == 10001:
       print(i)  
       break
    i += 1
    

