class MetricF1:

    def __init__(self, labels):
        self.labels = labels
        self.clear()

    def clear(self):
        self.tp = {l: 0 for l in self.labels}
        self.fp = {l: 0 for l in self.labels}
        self.fn = {l: 0 for l in self.labels}
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

    def update(self, preds, golds):
        for pred, gold in zip(preds, golds):
            for _, _, label in [x for x in pred if x in gold]:
                self.tp[label] += 1
                self.total_tp += 1
            for _, _, label in [x for x in pred if x not in gold]:
                self.fp[label] += 1
                self.total_fp += 1
            for _, _, label in [x for x in gold if x not in pred]:
                self.fn[label] += 1
                self.total_fn += 1

    def print(self, details=False):
        for label in self.labels:
            tp, fp, fn = self.tp[label], self.fp[label], self.fn[label]
            pr = tp / (tp + fp) if tp != 0 else 0.0
            re = tp / (tp + fn) if tp != 0 else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if tp != 0 else 0.0
            if details:
                print('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format(label, tp, fp, fn, pr, re, f1))

        print('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format('', self.total_tp, self.total_fp,
                                                                              self.total_fn, self.pr(), self.re(),
                                                                              self.f1()))

    def pr(self):
        return self.total_tp / (self.total_tp + self.total_fp) if self.total_tp != 0 else 0.0

    def re(self):
        return self.total_tp / (self.total_tp + self.total_fn) if self.total_tp != 0 else 0.0

    def f1(self):
        return 2 * self.total_tp / (2 * self.total_tp + self.total_fp + self.total_fn) if self.total_tp != 0 else 0.0


class MetricSpanNER:

    def __init__(self, task, verbose=False, labels=None, bio_labels=None):
        if labels is not None:
            self.labels = labels
        elif bio_labels is not None:
            self.labels = [label[2:] for label in bio_labels if label.startswith('B-')]
        else:
            raise BaseException('no labels')

        self.task = task
        self.evaluator = MetricF1(self.labels)
        self.iter = 0
        self.max_f1 = 0
        self.max_iter = 0
        self.verbose = verbose

    def step(self):
        self.evaluator.clear()
        self.iter += 1

    def update(self, pred, gold, metadata={}):
        self.evaluator.update(pred, gold)

    def update2(self, args, metadata={}):
        self.update(args['pred'], args['gold'], metadata=metadata)

    def print(self, dataset_name, details=False):
        f1 = self.evaluator.f1()

        if f1 > self.max_f1:
            self.max_f1 = f1
            self.max_iter = self.iter
        stall = self.iter - self.max_iter

        self.evaluator.print(self.verbose)

        print('EVAL-NER\t{}-{}\tcurr-iter: {}\tcurr-f1: {}\tmax-iter: {}\tmax-f1: {}\tstall: {}'.format(dataset_name,
                                                                                                        self.task,
                                                                                                        self.iter, f1,
                                                                                                        self.max_iter,
                                                                                                        self.max_f1,
                                                                                                        stall))

    def log(self, tb_logger, dataset_name):
        tb_logger.log_value('{}/f1'.format(dataset_name), self.evaluator.f1(), self.iter)
