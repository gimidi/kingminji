# 모델 자체를 함수에 집어 넣는 경우가 있을것임
# 재사용 재사용 재사용 재사용 재사용 재사용 재사용 재사용 재사용 재사용 재사용 재사용
# 클레스란? 우리 클레스 -> 책상, 컴퓨터, 각 학생들, 

def sum(a,b) :
    return(a+b)

print(sum(3,4))

### 곱셈, 나눗셈, 뺄셈 만드시오 ###

def mul1(a,b) :
    return(a*b)
def mul2(a,b,c) :
    return(a*b*c)
def div1(a,b) :
    return(a/b)
def sub1(a,b) :
    return(a-b)

a = 9 ; b = 3 ; c = 5
print(mul1(a,b))
#print(mul1(a,b,c)) #*** 오류/ mul1() takes 2 positional arguments but 3 were given

print(mul2(a,b,c))
print(div1(a,b)) 
print(sub1(a,b))

def sayYeah() : # 매개변수(파라미터)가 없는 함수
    return 'YeahYeah~'

print(sayYeah())

a = sayYeah() # 지금까지 모델 이렇게 다 넣었자너!! 
print(a)

