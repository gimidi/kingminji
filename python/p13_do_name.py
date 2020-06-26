import p11_car
import p12_tv

# 해당 파일을 실행함

# 운전하다
# car.py의 module 이름은  p11_car
# 시청하다
# tv.py의 module 이름은  p12_tv
# __name__이 __main__에서 -> 각 파일명으로 변화하였음
# 즉, 땡겨온 상태에서의 name 은 각 파일명이 된다

print("=============================")
print("do.py의 module 이름은 ",__name__)
print("=============================")


p11_car.drive()
p12_tv.watch()

# drive() 얘는 오류뜸요...