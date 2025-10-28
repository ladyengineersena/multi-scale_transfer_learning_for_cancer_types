#^fibonacci sayı dizisi ilk iki terimi 1 olan ve sonraki her terimi kendisinden önceki iki terimin toplamı olan dizidir.ilk 100 fibonacci sayısını yazın.
fibonacci_list = []
fibonacci_list.append(1)
fibonacci_list.append(1)

for i in range(2,100):
  fibonacci_list.append(fibonacci_list[i-2] + fibonacci_list[i-1])
print(fibonacci_list)