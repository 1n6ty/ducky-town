valid = False
elems = ['rock', 'papper', 'scissors']

fstInput = input('First input - ')
while not valid:
    if fstInput in elems:
        valid = True
    else:
        fstInput = input('Try again - ')

valid = False
secInput = input('Second input - ')
while not valid:
    if secInput in elems:
        valid = True
    else:
        secInput = input('Try again - ')

win = elems.index(fstInput) - elems.index(secInput)

if win == 2 or win == -1:
    print('Second player wins')
elif win == 1 or win == -2:
    print('First player wins')
else:
    print('tie')