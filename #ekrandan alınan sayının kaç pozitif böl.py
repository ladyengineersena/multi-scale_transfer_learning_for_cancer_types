#ekrandan alınan sayının kaç pozitif böleni olduğunu yaz
sayı = int(input("bir sayı giriniz:"))
positive_bölen = 0
for i in range(1,sayı+1):
    if sayı %i == 0:
        positive_bölen += 1
print(f"{sayı} sayısının {positive_bölen} tane bölen var")
