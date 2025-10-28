#100 basamaklı ilk fibonacci sayısını yaz
fibonacci_list = list()
fibonacci_list.append(1)
fibonacci_list.append(2)

index = 2
while True:
   fibonacci_list.append(fibonacci_list[index -2] + fibonacci_list[index -1])
   terim = fibonacci_list[index -2] + fibonacci_list[index -1]
   if len(str(terim)) == 100:
       print(terim)
       print(index)
       break
   index += 1
