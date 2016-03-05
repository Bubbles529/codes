#-*- coding:utf8 -*-
'''
约瑟夫环问题
'''


def GetLastOut(n, m):
    """
    编号n只老鼠从1~m报数，为m的出列，最后留下的编号
    """
    rats = list(range(1,n+1))

    curr_index = 0
    for i in range(n-1):
        curr_index += m - 1
        curr_index = curr_index % len(rats)
        remove_value = rats[curr_index]
        del rats[curr_index]
        #print('remove index:{}, value:{}, left:{}'.format(curr_index, remove_value, rats[curr_index:]+rats[:curr_index]))
    
    return rats[0]


def RescueGetLastOut(n, m):
    """
    递归方法，考虑去掉第一个编号之后，其排列顺序相当于[m+1 ... n 1 ... m-1]，与[1 ... n-1]相对应，可以直接进行变换
    """
    if n == 2:
        return 2 if m % 2 else 1
    else:
        v = RescueGetLastOut(n-1, m)
        #print('RescueGetLastOut({}, {}): {}'.format( n - 1, m, v))
        #v-1先转变为0~n-1便于%n，后续再+1变回1~n编号
        return (v - 1 + m) % n + 1



def GetLastOut(n, m):
    """
    根据递归方法而来的迭代方法，考虑去掉第一个编号之后，其排列顺序相当于[m+1 ... n 1 ... m-1]，与[1 ... n-1]相对应，可以直接进行变换
    """
    if n < 2:
        return 1
    
    live_rat = 1 if m % 2 else 0
    for i in range(3, n+1):
        live_rat = (live_rat + m) % i

    #前面是按照0~i-1编号，返回时改为1~n
    return live_rat + 1



def GetLastOutFor2(n):
    """
    针对2进行的特殊判断
    """
    import math
    log_v = int(math.log(n, 2))
    return 2*(n - 2**log_v) + 1

