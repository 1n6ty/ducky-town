from random import randint
length = int(input('Length - '))
print('Generated password -', "".join([chr(randint(48, 126)) for i in range(length)]))