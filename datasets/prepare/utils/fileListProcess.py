import multiprocessing



class FileListProcess():
    def __init__(self, func, file_list, is_multi=False):
        '''
        :param func(method):将要执行的函数
        :param file_list(list[str]): 需要处理的文件的列表
        :param is_multi(bool): 如果是True，那么将会调用多进程方式实现，否则将为单进程方式实现
        '''
        self.worker_num=16
        self.is_multi=is_multi
        self.func=func
        self.file_list=file_list

    def non_multi_process(self,):
        '''
        非多进程实现
        :return:
        '''
        for i,image_file in enumerate(self.file_list):
            self.func(image_file)

    def multi_process(self,):
        '''
        多进程实现
        :return:
        '''
        pool = multiprocessing.Pool(processes=self.worker_num)
        for file_name in self.file_list:
            pool.apply_async(self.func, (file_name,))
        pool.close()
        pool.join()

    def process(self,):
        if self.is_multi:
            self.multi_process()
        else:
            self.non_multi_process()