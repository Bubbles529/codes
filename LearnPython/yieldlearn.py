def test_for_yield_once():
    def t():
        return list(range(5))
    x = t()
    ret = sum(x) / max(x)
    print(ret)

    def t2():
        for i in range(5):
            yield i
    x = t2()
    ret = sum(x) / max(x)
    print(ret)


def  stopiteration_test():
    def f():
        yield 2
    g = f()
    for i in range(5):
        try:
            print(next(g))
        except Exception as e:
            print(e.__class__.__name__)


def yield_with_exception():
    def f():
        raise StopIteration()
        yield 
    def g():
        raise Exception('xx')
        yield

    for i in f():
        pass
    for i in g():
        pass
    
        


def yield_with_return():
    def f():
        return 1
        yield 2
    def g():
        yield 2
        return 4
    for i in f():
        print('return before yield', i)
    for i in g():
        print('yield before before', i)
    print('return before yield, what if return value', f())


def yield_test_send():
    def f():
        print('inner before')
        y = yield
        print('inner mid')
        z = yield y+1
        print('inner end z: ')
        yield z+1
    g = f()
    print('out first next', next(g))
    print('out send 1', g.send(5))
    print('out send 2', g.send(10))


def test_close():
    def f():
        print('inner')
        yield 2
        yield 3

    g = f()
    for i in range(4):
        try:
            print('g.close()',g.close())
            print('next(g):',next(g))
        except Exception as e:
            print('except:',e.__class__.__name__)


def test_throw():
    def g():
        print('inner 1')
        yield 2
        yield 3
        yield 4
    ge = g()
    for i in range(4):
        try:
            ge.throw(Exception, 'throw test')
        except Exception as e:
            print(e, e.__class__.__name__)
    next(ge)


def test_contextmanager():
    from contextlib import contextmanager
    @contextmanager
    def f():
        print('before')
        yield 2
        print('end')
    with f() as k:
        print('k', k)
        


if __name__ == '__main__':
    test_contextmanager()
