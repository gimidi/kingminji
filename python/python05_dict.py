#3. 딕셔너리 # 중복 x
# {key : value}
# key는 index 같은 느낌이다.
a = {11: 'hi', 22:'minji','이야':'호!'}
print(a)
print(a[11]) # hi
print(a[22]) # minji
print(a['이야']) # 호!

# 딕셔너리 요소 삭제
del a[11]   # 11:'hi'를 쌍으로 지운다
print(a)

del a[22]
print(a)

a = {1:'a', 1:'b', 1:'c'}
print(a) # {1: 'c'} 가장 마지막 요소만 나온다 / 그 뒤 요소들이 덮어썼다고 생각하면 된다.

a = {1:'a', 2:'a', 3:'a'}
print(a)

a = {'name':'yun', 'phone':'010', 'birth':'0511'}
print(a.keys()) #.ddd에서 ()안붙는게 있었나? 왜 자꾸 안붙이게되지..?
print(a.values())
print(a.get('name'))
print(a['name'])
