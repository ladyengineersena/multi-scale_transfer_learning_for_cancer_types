#Find the largest palindrom made from the product of two 3-digit numbers.

def check_palindrome(x):
    str_number = str(x)
    reverse_number = str_number[::-1]
    if str_number == reverse_number:
       return True
    else:
       return False

big_palindrome = 0
#i 100 iken j 101 , 102... olucak.
for i in range(100,1000):
   for j in range(100,1000):
       if check_palindrome (i*j) and i*j > big_palindrome:
           big_palindrome = i*j
print(big_palindrome)

