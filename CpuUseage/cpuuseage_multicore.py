""" 
控制CPU使用曲线（多核版本） 
"""  
import timeit  
import time  
import os  
import multiprocessing  
import math  
  
  
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
SLEEP_TIME = timeit.timeit("run_cpu()", setup="from "+__name__+" import run_cpu", number=TEST_TIME_NUMBERS)/TEST_TIME_NUMBERS  
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
                  
  
def run_sin(queues):  
    """ 
    使用sin函数 
    """  
    i = 0  
    while True:  
        i = i + 1  
        use_age = 50 + math.sin(i/5)*50  
        for q in queues:  
            q.put(use_age)  
        run_for_1s(use_age)  
  
  
def run_second_process(queue, sleep_time):  
    """ 
    其他核的执行函数 
    """  
    all_times = 1 / sleep_time  
      
    while True:  
        percent = queue.get()  
        cpu_times = all_times * percent / 100  
          
        for i in range(int(cpu_times)):  
            run_cpu()  
  
def main():  
    core_num = os.cpu_count()  
    queues = []  
    if core_num:  
        for i in range(core_num-1):  
            q = multiprocessing.Queue()  
            queues.append(q)  
            p = multiprocessing.Process(target=run_second_process, args=(q, SLEEP_TIME))  
            p.start()  
    run_sin(queues)  
  
  
if __name__ == '__main__':  
    main()  
