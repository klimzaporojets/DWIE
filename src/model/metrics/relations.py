def to_pairwise(rels):
    out = []
    for src_cluster, dst_cluster, rel in rels:
        for src in src_cluster:
            for dst in dst_cluster:
                out.append((src, dst, rel))
    return set(out)


def to_pairs(src_cluster, dst_cluster, rel):
    pairs = []
    for src in src_cluster:
        for dst in dst_cluster:
            pairs.append((src, dst, rel))
    return set(pairs)


def captions(cluster, tokens):
    return [' '.join(tokens[begin:(end + 1)]) for begin, end in cluster]


# concept wise relation
class MetricConceptRelationSoftF1:

    def __init__(self, name, labels, verbose):
        self.task = name
        self.labels = labels
        self.iter = 0
        self.max_f1 = 0
        self.max_iter = 0
        self.verbose = verbose

    def step(self):
        self.p_tps = {l: 0.0 for l in self.labels}
        self.p_fps = {l: 0.0 for l in self.labels}
        self.r_tps = {l: 0.0 for l in self.labels}
        self.r_fns = {l: 0.0 for l in self.labels}
        self.iter += 1

    def add(self, pred, gold):
        P = to_pairwise(pred)
        G = to_pairwise(gold)

        for src_cluster, dst_cluster, rel in pred:
            pairs = to_pairs(src_cluster, dst_cluster, rel)
            if len(pairs) > 0:
                tp = len(pairs & G) / len(pairs) if len(pairs) != 0 else 0.0
                fp = 1.0 - tp

                self.p_tps[rel] += tp
                self.p_fps[rel] += fp

        for src_cluster, dst_cluster, rel in gold:
            pairs = to_pairs(src_cluster, dst_cluster, rel)
            if len(pairs) > 0:
                tp = len(pairs & P) / len(pairs) if len(pairs) != 0 else 0.0
                fn = 1.0 - tp

                self.r_tps[rel] += tp
                self.r_fns[rel] += fn

    def update2(self, args, metadata={}):
        for batch, (pred, gold, identifier, tokens) in enumerate(
                zip(args['pred'], args['gold'], metadata['identifiers'], metadata['tokens'])):

            self.add(pred, gold)

    def print(self, dataset_name, details):
        total_p_tp, total_p_fp, total_r_tp, total_r_fn = 0, 0, 0, 0

        for label in self.labels:
            p_tp, p_fp = self.p_tps[label], self.p_fps[label]
            r_tp, r_fn = self.r_tps[label], self.r_fns[label]
            pr = p_tp / (p_tp + p_fp) if p_tp != 0 else 0.0
            re = r_tp / (r_tp + r_fn) if r_tp != 0 else 0.0
            f1 = 2.0 * pr * re / (pr + re) if pr * re != 0.0 else 0.0

            if self.verbose:
                print(
                    '{:24}    {:5.1f} / {:5.1f} = {:6.5f}    {:5.1f} / {:5.1f} = {:6.5f}    {:6.5f}'.format(label, p_tp,
                                                                                                            p_fp, pr,
                                                                                                            r_tp, r_fn,
                                                                                                            re, f1))

            total_p_tp += p_tp
            total_p_fp += p_fp
            total_r_tp += r_tp
            total_r_fn += r_fn

        total_pr = total_p_tp / (total_p_tp + total_p_fp) if total_p_tp != 0 else 0.0
        total_re = total_r_tp / (total_r_tp + total_r_fn) if total_r_tp != 0 else 0.0
        total_f1 = 2.0 * total_pr * total_re / (total_pr + total_re) if total_pr * total_re != 0.0 else 0.0

        print('{:24}    {:5.1f} / {:5.1f} = {:6.5f}    {:5.1f} / {:5.1f} = {:6.5f}    {:6.5f}'.format('', total_p_tp,
                                                                                                      total_p_fp,
                                                                                                      total_pr,
                                                                                                      total_r_tp,
                                                                                                      total_r_fn,
                                                                                                      total_re,
                                                                                                      total_f1))

        self.f1 = total_f1
        if self.f1 > self.max_f1:
            self.max_f1 = self.f1
            self.max_iter = self.iter
        stall = self.iter - self.max_iter

        print('EVAL-REL-SOFT\t{}-{}*\tcurr-iter: {}\tcurr-f1: {}\tmax-iter: {}\tmax-f1: {}\tstall: {}'.format(
            dataset_name, self.task, self.iter, self.f1, self.max_iter, self.max_f1, stall))

    def log(self, tb_logger, dataset_name):
        tb_logger.log_value('{}/{}-f1'.format(dataset_name, self.task), self.f1, self.iter)


# concept wise relation
class MetricConceptRelationToMentionsF1:

    def __init__(self, name, labels, verbose):
        self.task = name
        self.labels = labels
        self.iter = 0
        self.max_f1 = 0
        self.max_iter = 0
        self.verbose = verbose

    def step(self):
        self.tps = {l: 0 for l in self.labels}
        self.fps = {l: 0 for l in self.labels}
        self.fns = {l: 0 for l in self.labels}
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.iter += 1

    def add(self, pred, gold):
        P = to_pairwise(pred)
        G = to_pairwise(gold)

        for _, _, rel in P & G:
            self.tps[rel] += 1
            self.total_tp += 1
        for _, _, rel in P - G:
            self.fps[rel] += 1
            self.total_fp += 1
        for _, _, rel in G - P:
            self.fns[rel] += 1
            self.total_fn += 1

    def update2(self, args, metadata={}):
        for batch, (pred, gold, identifier, tokens) in enumerate(
                zip(args['pred'], args['gold'], metadata['identifiers'], metadata['tokens'])):

            self.add(pred, gold)

    def print(self, dataset_name, details=False):
        for label in self.labels:
            tp, fp, fn = self.tps[label], self.fps[label], self.fns[label]
            pr = tp / (tp + fp) if tp != 0 else 0.0
            re = tp / (tp + fn) if tp != 0 else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if tp != 0 else 0.0
            if self.verbose:
                print('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format(label, tp, fp, fn, pr, re, f1))

        print('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format('', self.total_tp, self.total_fp,
                                                                              self.total_fn, self.pr(), self.re(),
                                                                              self.f1()))

        self.last_f1 = self.f1()
        if self.last_f1 > self.max_f1:
            self.max_f1 = self.last_f1
            self.max_iter = self.iter
        stall = self.iter - self.max_iter

        print(
            'EVAL-REL-C2M\t{}-{}*\tcurr-iter: {}\tcurr-f1: {}\tmax-iter: {}\tmax-f1: {}\tstall: {}'.format(dataset_name,
                                                                                                           self.task,
                                                                                                           self.iter,
                                                                                                           self.last_f1,
                                                                                                           self.max_iter,
                                                                                                           self.max_f1,
                                                                                                           stall))

    def pr(self):
        return self.total_tp / (self.total_tp + self.total_fp) if self.total_tp != 0 else 0.0

    def re(self):
        return self.total_tp / (self.total_tp + self.total_fn) if self.total_tp != 0 else 0.0

    def f1(self):
        return 2 * self.total_tp / (2 * self.total_tp + self.total_fp + self.total_fn) if self.total_tp != 0 else 0.0

    def log(self, tb_logger, dataset_name):
        tb_logger.log_value('{}/{}-tomention-f1'.format(dataset_name, self.task), self.last_f1, self.iter)


# standard f1 scoring for relations computed on predicted span
class MetricSpanRelationF1x:

    def __init__(self, name, labels, verbose):
        self.task = name
        self.labels = labels
        self.iter = 0
        self.max_f1 = 0
        self.max_iter = 0
        self.verbose = verbose

    def step(self):
        self.tps = {l: 0 for l in self.labels}
        self.fps = {l: 0 for l in self.labels}
        self.fns = {l: 0 for l in self.labels}
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.iter += 1

    def add(self, pred, gold):
        P = set(pred)
        G = set(gold)

        for _, rel, _ in P & G:
            self.tps[rel] += 1
            self.total_tp += 1
        for _, rel, _ in P - G:
            self.fps[rel] += 1
            self.total_fp += 1
        for _, rel, _ in G - P:
            self.fns[rel] += 1
            self.total_fn += 1

    def update2(self, args, metadata={}):
        for batch, (pred, gold, identifier, tokens) in enumerate(
                zip(args['span-rel-pred'], args['span-rel-gold'], metadata['identifiers'], metadata['tokens'])):
            self.add(pred, gold)

    def print(self, dataset_name, details=False):
        for label in self.labels:
            tp, fp, fn = self.tps[label], self.fps[label], self.fns[label]
            pr = tp / (tp + fp) if tp != 0 else 0.0
            re = tp / (tp + fn) if tp != 0 else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if tp != 0 else 0.0
            if self.verbose:
                print('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format(label, tp, fp, fn, pr, re, f1))

        print('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format('', self.total_tp, self.total_fp,
                                                                              self.total_fn, self.pr(), self.re(),
                                                                              self.f1()))

        self.last_f1 = self.f1()
        if self.last_f1 > self.max_f1:
            self.max_f1 = self.last_f1
            self.max_iter = self.iter
        stall = self.iter - self.max_iter

        print('EVAL-REL-SPAN\t{}-{}*\tcurr-iter: {}\tcurr-f1: {}\tmax-iter: {}\tmax-f1: {}\tstall: {}'.format(
            dataset_name, self.task, self.iter, self.last_f1, self.max_iter, self.max_f1, stall))

    def pr(self):
        return self.total_tp / (self.total_tp + self.total_fp) if self.total_tp != 0 else 0.0

    def re(self):
        return self.total_tp / (self.total_tp + self.total_fn) if self.total_tp != 0 else 0.0

    def f1(self):
        return 2 * self.total_tp / (2 * self.total_tp + self.total_fp + self.total_fn) if self.total_tp != 0 else 0.0

    def log(self, tb_logger, dataset_name):
        tb_logger.log_value('{}/{}-f1'.format(dataset_name, self.task), self.last_f1, self.iter)
