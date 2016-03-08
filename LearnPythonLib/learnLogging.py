def test_1():
    import logging  
    logging.debug('debug message')  
    logging.info('info message')  
    logging.warning('warning message')  
    logging.error('error message')  
    logging.critical('critical message')

def test_2():
    import logging
    logger = logging.getLogger('root.test')
    logger.setLevel(logging.CRITICAL)
    logger.debug('debug message')  
    logger.info('info message')  
    logger.warning('warning message')  
    logger.error('error message')  
    logger.critical('critical message')

def test_3():
    import logging
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
    ch.setFormatter(formatter) 
    logger1 = logging.getLogger('root.test')
    logger1.addHandler(ch)
    logger2 = logging.getLogger('root.test.xxx')
    logger2.addHandler(ch)
    logger2.setLevel(logging.CRITICAL)
    logger2.critical('critical message')

if __name__ == '__main__':
    test_3()
