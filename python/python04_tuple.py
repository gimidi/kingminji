#2. 튜플 , 소괄호
# 리스트와 거의 같으나, 삭제, 수정이 안된다. (바뀌지 않는 값)

a = (1,2,3)
b = 1,2,3
print(type(a)) # <class 'tuple'>
print(type(b)) # <class 'tuple'>

# a.remove(2)
# print(a) AttributeError: 'tuple' object has no attribute 'remove'

print(a + b)  # (1, 2, 3, 1, 2, 3) 개체안은 수정이 안된다
print(a * 3)  # (1, 2, 3, 1, 2, 3, 1, 2, 3)

print(a - 3) # TypeError: unsupported operand type(s) for -: 'tuple' and 'int' / 요소를 바꿀 수 없다...!
