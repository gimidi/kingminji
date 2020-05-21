# while은 안해주려나..

a = {'name':'yun', 'phone':'010', 'birth':'0511'}

for i in a.keys() :
    print(i)

a = [1,2,3,4,5,6,7,8,9,10]
for i in a :
    i = i*i
    print(i)
print('minji')

# while도 해주네? 만세 !!
# while  참일 동안 계속 돈다.   아니 왜 말 하다말아 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ

if 0 :              # False 얘만 False임!
    print('True')
else :
    print('False')

if 1 :              # True
    print('True')
else :
    print('False')

if 3 :              # True
    print('True')
else :
    print('False')

'''
>< == != >= <=

'''

### 조건 연산자 ( and / or / not )
money = 30000
card = 1
if money >= 30000 or card == 1:
    print('한우')
else :
    print('라면..')

점수 = [90,25,67,45,80]
num = 0
for i in 점수 :
    if i >= 60 :
        print('합격')
        num += 1
print('합격자 수는 {0}명 입니다'.format(num))


# break, continue
####################### break ###########################
num = 0
for i in 점수 :
    if i < 30 :
        break # for문 자체를 중단한다 (break문과 가장 가까운 반복문에서 빠져나옴)

    if i >= 60 :
        print('합격')
        num += 1
print('합격자 수는 {0}명 입니다'.format(num))

####################### continue ###########################
num = 0
for i in 점수 :
    if i < 60 :
        continue # 하단 실행 안하고 for문으로 돌아간다

    if i >= 60 :
        print('합격')
        num += 1
print('합격자 수는 {0}명 입니다'.format(num))