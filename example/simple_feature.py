from feature.simple import SimpleFeature
from dataset.dataset import SemEvalDataSet

simple_feature = SimpleFeature("SimpleFeature")
features = simple_feature.transform(SemEvalDataSet)