from module_classes import ProcessingModule
import torch
from Fastformer import Model
from SiameseDualEncoder import SDE
import torch.optim as optim
import numpy as np
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import ndcg_score


def reciprocal_rank(true, hat):
    sorted_ranks = sorted(enumerate(hat), key=lambda t: t[1])
    ground_truth_position = -1
    for i, x in sorted_ranks:
        try:
            if i == list(true).index(1):
                ground_truth_position = i+1
                break
        except ValueError: # '1' not in 'true'
            break
    if ground_truth_position != -1:
        return 1/ground_truth_position
    else:
        return 0


class TrainingModule(ProcessingModule):
    def __init__(self, train_config, model_config, pipeline_config) -> None:
        super().__init__(train_config, model_config)
        self.pipeline_config = pipeline_config

        # FF only
        self.SELECTED_METRIC = "mrr"
        self.METRICS = {
            "pcc": (lambda true, hat: pearsonr(true, hat).statistic, "Pearson correlation coefficient"),
            "ndcg10": (lambda true, hat: ndcg_score(true, hat, k=10), "Normalized Discounted Cumulative Gain @ 10"),
            "mrr": (lambda true, hat: reciprocal_rank(true, hat), "Reciprocal Rank")
        }

    def __FF_trainer(self, text, label, train_idx, test_idx):
        model = Model(self.model_config)

        optimizer = optim.Adam(
            [{"params": model.parameters(), "lr": self.train_config.learning_rate}]
        )

        model.cuda()

        # training
        for i in range(self.train_config.num_epochs):
            loss = 0.0
            metric_score = 0.0
            for num_batch in range(len(train_idx) // self.train_config.batch_size):

                log_ids = text[train_idx][
                    num_batch * self.train_config.batch_size
                    : num_batch * self.train_config.batch_size + self.train_config.batch_size,
                    : self.train_config.tokenizer_max_length,
                ]
                targets = label[train_idx][
                    num_batch * self.train_config.batch_size 
                    : num_batch * self.train_config.batch_size + self.train_config.batch_size
                ]

                log_ids = torch.LongTensor(log_ids).cuda(0, non_blocking=True)
                targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)
                bz_loss, y_hat = model(log_ids, targets)

                loss += bz_loss.data.float()

                this_score = self.METRICS[self.SELECTED_METRIC][0](
                    targets.to("cpu").detach().numpy().tolist(),
                    y_hat.to("cpu").detach().numpy().tolist()
                )
                # scipy metrics sometimes return nan
                if np.isnan(this_score):
                    this_score = 0
                metric_score += this_score

                unified_loss = bz_loss
                optimizer.zero_grad()
                unified_loss.backward()
                optimizer.step()

                if num_batch % 100 == 0:
                    print(
                        "[TRAIN SET] Epoch: {}, Samples: 0-{}, train_loss: {:.5f}, {}: {:.5f}".format(
                            i+1,
                            self.train_config.batch_size + (num_batch * self.train_config.batch_size), 
                            loss.data / (num_batch + 1), # mean over all batches processed so far
                            self.SELECTED_METRIC,
                            metric_score / (num_batch + 1) # mean over all batches processed so far
                        )
                    )
            
            # evaluation
            model.eval()
            y_hat_all = []
            loss2 = 0.0
            for num_batch in range(len(test_idx) // self.train_config.batch_size + 1):

                log_ids = text[test_idx][
                    num_batch * self.train_config.batch_size 
                    : num_batch * self.train_config.batch_size + self.train_config.batch_size, 
                    : self.train_config.tokenizer_max_length
                ]
                targets = label[test_idx][
                    num_batch * self.train_config.batch_size 
                    : num_batch * self.train_config.batch_size + self.train_config.batch_size
                ]
                log_ids = torch.LongTensor(log_ids).cuda(0, non_blocking=True)
                targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)

                bz_loss2, y_hat2 = model(log_ids, targets)

                loss2 += bz_loss2.data.float()
                y_hat_all += y_hat2.to("cpu").detach().numpy().tolist()

            y_true = label[test_idx]
            print("[TEST SET] {} after epoch {}: {:.5f} (loss: {:.5})\n".format(
                self.METRICS[self.SELECTED_METRIC][1],
                i+1,
                self.METRICS[self.SELECTED_METRIC][0](y_true, y_hat_all),
                loss2
            ))

            # reset model state
            model.train()

    def __SDE_trainer(self, query, document, target, train_idx, test_idx):
        model = SDE(self.model_config)

        optimizer = optim.Adam(
            [{"params": model.parameters(), "lr": self.train_config.learning_rate}]
        )

        model.cuda()

        # training
        for i in range(self.train_config.num_epochs):
            loss = 0.0
            metric_score = 0.0
            for num_batch in range(len(train_idx) // self.train_config.batch_size):

                qry_logids = query[train_idx][
                    num_batch * self.train_config.batch_size
                    : num_batch * self.train_config.batch_size + self.train_config.batch_size,
                    : self.train_config.tokenizer_max_length,
                ]
                doc_logids = document[train_idx][
                    num_batch * self.train_config.batch_size
                    : num_batch * self.train_config.batch_size + self.train_config.batch_size,
                    : self.train_config.tokenizer_max_length,
                ]
                targets = target[train_idx][
                    num_batch * self.train_config.batch_size 
                    : num_batch * self.train_config.batch_size + self.train_config.batch_size
                ]
                qry_logids = torch.LongTensor(qry_logids).cuda(0, non_blocking=True)
                doc_logids = torch.LongTensor(doc_logids).cuda(0, non_blocking=True)
                targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)

                bz_loss, y_hat = model(qry_logids, doc_logids, targets)

                loss += bz_loss.data.float()
                
                this_score = reciprocal_rank(
                    targets.to("cpu").detach().numpy().tolist(),
                    y_hat.to("cpu").detach().numpy().tolist()
                )
                metric_score += this_score

                unified_loss = bz_loss
                optimizer.zero_grad()
                unified_loss.backward()
                optimizer.step()

                if num_batch % 100 == 0:
                    print(
                        "[TRAIN SET] Epoch: {}, Samples: 0-{}, train_loss: {:.5f}, mrr: {:.5f}".format(
                            i+1,
                            self.train_config.batch_size + (num_batch * self.train_config.batch_size), 
                            loss.data / (num_batch + 1), # mean over all batches processed so far
                            metric_score / (num_batch + 1) # mean over all batches processed so far
                        )
                    )

            # evaluation
            model.eval()
            all_y_hat = []
            per_batch_metric = []
            loss2 = 0.0
            
            for num_batch in range(len(test_idx) // self.train_config.batch_size + 1):

                qry_logids = query[test_idx][
                    num_batch * self.train_config.batch_size
                    : num_batch * self.train_config.batch_size + self.train_config.batch_size,
                    : self.train_config.tokenizer_max_length,
                ]
                doc_logids = document[test_idx][
                    num_batch * self.train_config.batch_size
                    : num_batch * self.train_config.batch_size + self.train_config.batch_size,
                    : self.train_config.tokenizer_max_length,
                ]
                targets = target[test_idx][
                    num_batch * self.train_config.batch_size 
                    : num_batch * self.train_config.batch_size + self.train_config.batch_size
                ]
                qry_logids = torch.LongTensor(qry_logids).cuda(0, non_blocking=True)
                doc_logids = torch.LongTensor(doc_logids).cuda(0, non_blocking=True)
                targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)

                bz_loss2, y_hat2 = model(qry_logids, doc_logids, targets)

                loss2 += bz_loss2.data.float()
                y_hat2 = y_hat2.to("cpu").detach().numpy().tolist()
                all_y_hat += y_hat2
                per_batch_metric.append(reciprocal_rank(targets, y_hat2))

            print("[TEST SET] Mean Reciprocal Rank after epoch {}: {:.5f} (loss: {:.5})\n".format(
                i+1,
                np.mean(per_batch_metric),
                loss2
            ))

            # reset model state
            model.train()

    def execute(self, args):

        if self.pipeline_config.model == "FF":
            text, label = args

            print("creating split indices...")
            index = np.arange(len(label))
            breakpoint = int(np.ceil(self.train_config.train_split_size * len(index)))
            train_idx = index[:breakpoint]
            if not self.train_config.batch_padding:
                np.random.shuffle(train_idx)
            test_idx = index[breakpoint:]
            print(" - done")

            print(f"training {self.pipeline_config.model}...")
            self.__FF_trainer(text, label, train_idx, test_idx)
            print(" - done")

        elif self.pipeline_config.model == "SDE":
            query, document, target = args

            print("creating split indices...")
            index = np.arange(len(target))
            breakpoint = int(np.ceil(self.train_config.train_split_size * len(index)))
            train_idx = index[:breakpoint]
            if not self.train_config.batch_padding:
                np.random.shuffle(train_idx)
            test_idx = index[breakpoint:]
            print(" - done")

            print(f"training {self.pipeline_config.model}...")
            self.__SDE_trainer(query, document, target, train_idx, test_idx)
            print(" - done")

        else:
            raise ValueError(f"Model {self.pipeline_config.model} is not implemented!")
        