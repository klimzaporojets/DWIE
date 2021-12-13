
class MetricObjective:

    def __init__(self, task):
        self.task = task
        self.clear()
        
    def clear(self):
        self.total = 0
        self.iter = 0
    
    def step(self):
        self.total = 0
        self.iter += 1

    def update(self, logits, targets, args, metadata={}):
        self.total += args['obj']

    def update2(self, args, metadata={}):
        self.total += args['loss']

    def print(self, dataset_name, details=False):
        print('EVAL-OBJ\t{}-{}\tcurr-iter: {}\tobj: {}'.format(dataset_name, self.task, self.iter, self.total))

    def log(self, tb_logger, dataset_name):
        tb_logger.log_value('{}/{}-obj'.format(dataset_name, self.task), self.total, self.iter)
