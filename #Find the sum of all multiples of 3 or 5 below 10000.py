#Find the sum of all multiples of 3 or 5 below 1000
def check(x):
   if x %3 == 0 or x%5 == 0:
      return True
   else:
      return False
sum = 0
for i in range(1,1000):
    if check(i) == True:
      sum +=i
print(sum)