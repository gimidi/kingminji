# import p31_sample
from p31_sample import test

x = 222

def main_func() :
    print('x : ',x)

# p31_sample.test() # 111 -> 해당 함수는 해당 파일 범위안에서만 실행된다는 소리임
test()  # 111 -> from import 함수명으로 아예 임포트해서 쓸 수도 있다 (사용목적에 따라서 가장 효율적인 것 기획하기)
main_func() # 222