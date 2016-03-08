
def old_use():
    iterable = [1,2,2,3,3,4,4,4,4,4,5,5,5,5,6]
    counter = {}
    for i in iterable:
        if i not in counter:
            counter[i] = 1
        else:
            counter[i] += 1
    top2 = sorted(counter.items(), key=lambda i:i[1], reverse=True)[:2]
    print(top2)

def counter_create():
    from collections import Counter
    counter = Counter()
    
    iterable = [1,2,2,3,3,4,4,4,4,4,5,5,5,5,6]
    counter_a = Counter(iterable)
    print(counter_a.most_common(2))

    counter_b = Counter({4:5,5:4})
    print(counter_b.most_common(2))

    counter_c= Counter(a=5,b=4)
    print(counter_c.most_common(2))

def visit():
    from collections import Counter
    #访问单个元素
    iterable = [1,2,2,3,3,4,4,4,4,4,5,5,5,5,6]
    counter_a = Counter(iterable)
    print(counter_a[4])
    print(counter_a[8])  #不存在返回0

    #访问次数最多
    print(counter_a.most_common(2)) #参数为空则返回所有

def update():
    from collections import Counter
    #增加
    iterable = [1,2,2,3,3,4,4,4,4,4,5,5,5,5,6]
    counter_a = Counter(iterable)
    counter_a.update(iterable)
    print(counter_a[4])
    counter_a.update(counter_a)
    print(counter_a[4])

    #减少
    counter_b = Counter(iterable)
    counter_a.subtract(counter_b)
    print(counter_a[4])

def delete():
    from collections import Counter
    #删除
    iterable = [1,2,2,3,3,4,4,4,4,4,5,5,5,5,6]
    counter_a = Counter(iterable)
    del counter_a[4]
    print(counter_a[4])


def elements():
    from collections import Counter
    #删除
    iterable = [1,2,2,3,3,4,4,4,4,4,5,5,5,5,6]
    counter_a = Counter(iterable)
    print(list(counter_a.elements()))

def deleteNegtibe():
    from collections import Counter
    #删除
    iterable = [1,2,2,3,3,4,4,4,4,4,5,5,5,5,6]
    counter_a = Counter(iterable)
    counter_a[9] = -1
    counter_a[10] = 0
    print(counter_a.items())
    counter_a += Counter()
    print(counter_a.items())


    


if __name__ == '__main__':
    elements()
    deleteNegtibe()
