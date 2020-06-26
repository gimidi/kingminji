from machine.car import drive
from machine.tv import watch

drive()
watch()

from machine import car
from machine import tv

car.drive()
tv.watch()

print('======================================')

from machine.test.car import drive
from machine.test.tv import watch

drive()
watch()

from machine.test import car
from machine.test import tv

car.drive()
tv.watch()

from machine import test
# from machine import tv

test.car.drive()
test.tv.watch()

# 만든 폴더를 import해서 쓸 수 있다. 그래서 vc를 쓰는게 좋은것임