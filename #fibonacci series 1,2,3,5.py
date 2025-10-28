#fibonacci series 1,2,3,5
fibonacci = list()
fibonacci.append(1)
fibonacci.append(2)
i = 2
while True:
 if fibonacci[i-1] + fibonacci[i-2] <400000:
    fibonacci.append(fibonacci[i-1]+fibonacci[i-2])
    i +=1
 else:
     break
sum = 0
for i in fibonacci:
    if i %2 == 0:
        sum += i
print(sum)
