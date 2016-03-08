#-*- coding:utf8 -*-
'''
获得函数的名字
'''

def show_doc_first(f):
    '用以显示函数的doc字符串'
    def new_f(*a, **ka):
        print(f.__doc__)
        return f(*a, **ka)
    return new_f

    
@show_doc_first    
def get_name_out():
    '''函数外边获得函数的名字'''
    def fun_name():
        pass

    z = fun_name
    print(z.__name__)
    print(getattr(z,'__name__'))


@show_doc_first    
def get_name_by_frame():
    '''函数内部通过sys._getframe().f_code.co_name获得名字'''
    def fun_name():
        import sys
        print(sys._getframe().f_code.co_name)

    z = fun_name
    z()


@show_doc_first
def get_name_by_dec():
    '通过装饰器将名字作为参数传入函数'
    def dec_name(f):
        name = f.__name__
        def new_f(*a, **ka):
            return f(*a, __name__ = name, **ka)
        return new_f

    @dec_name
    def fun_name(x, __name__):
        print(__name__)
        
    z = fun_name
    z(1)


@show_doc_first
def get_name_by_inspect():
    '通过inspect获取名字'
    import inspect
    def fun_name():
        print(inspect.stack()[0][3])
    z = fun_name
    z()


if __name__ == '__main__':
    get_name_out()
    get_name_by_frame()
    get_name_by_dec()
    get_name_by_inspect()
