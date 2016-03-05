#-*- coding: utf8 -*-  
""" 
控制CPU使用曲线 
"""  
import timeit  
import time  
  
  
def run_cpu():  
    """ 
    消耗CPU运算时执行的程序 
    """  
    x = [_ for _ in range(RUN_RANGE_NUMBER)]  
  
  
def idle_cpu():  
    """ 
    保持CPU处于IDLE的Sleep程序 
    """  
    time.sleep(SLEEP_TIME)  
  
TEST_TIME_NUMBERS = 100  
RUN_RANGE_NUMBER = 100000     
SLEEP_TIME = timeit.timeit("run_cpu()", setup="from __main__ import run_cpu", number=TEST_TIME_NUMBERS)/TEST_TIME_NUMBERS  
print("测试运行100次，每次执行时间为：{}S".format(SLEEP_TIME))  
  
      
def run_for_1s(percent):  
    """ 
    1s中按照percent比例分别执行程序或者休眠 
    """  
    all_times = 0.33 / SLEEP_TIME  
    cpu_times = all_times * percent / 100  
          
    for _ in range(3):  
        for i in range(int(all_times)):  
            if i < cpu_times:  
                run_cpu()  
            else:  
                idle_cpu()  
                  
  
def run_sin():  
    """ 
    使用sin函数 
    """  
    import math  
    i = 0  
    while True:  
        i = i + 1  
        use_age = 50 + math.sin(i/5)*50  
        run_for_1s(use_age)  
  
  
if __name__ == '__main__':  
    run_sin()  
