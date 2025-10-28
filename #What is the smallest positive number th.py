#What is the smallest positive number that is evenly divisible all of numbers from 1 to 20
#1 ve 20 arasındaki tüm çift sayılara bölünen en küçük pozitif sayı?
import math
import functools
def gcd (x,y):
    return math.gcd(x,y)
#ekok
def lcm (x,y):
    return(x*y)//gcd(x,y)

liste = range(1,21)
result = functools.reduce(lcm,liste)
print(result)
