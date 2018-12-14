from dataset.dataset import SemEvalDataSet

for row in SemEvalDataSet.iterate_train():
    print(row)
