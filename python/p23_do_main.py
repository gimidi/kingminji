import p21_car
import p22_tv


# 땡겨온 상태에서의 name 은 main이 아닌, 각 파일명이기 때문에 실행되지 않고 그냥 불러오기만 한다.

print("=============================")
print("do.py의 module 이름은 ",__name__)
print("=============================")

p21_car.drive()
p22_tv.watch()