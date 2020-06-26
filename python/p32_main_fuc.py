import p31_sample

x = 222

def main_func() :
    print('x : ',x)

p31_sample.test() # 111 -> 해당 함수는 해당 파일 범위안에서만 실행된다는 소리임

main_func() # 222