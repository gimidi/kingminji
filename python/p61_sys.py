import sys #시스템
print(sys.path)
# 첫번째가 지금 경로
# 오 'C:\\Users\\bitcamp\\anaconda3' 이것두 있네~
# 여기서 보이는 경로가 파이썬 쓸때마다 땡겨올 수 있는 폴더 (패스가 걸려있는 폴더)
# 즉, 상시적으로 쓰고 싶은 함수가 있으면 여기다가 넣어놓으면 됨

from test_import import p62_import # 파일 자체를 import한거라서 print문 출력됨
p62_import.sum2()

print('===========================================')

from test_import.p62_import import sum2 # 이건 함수까지 내려가서 함수 자체를 가져온거라 함수만 가져옴
sum2()

