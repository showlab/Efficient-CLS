from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn as nn


class EnsembleTSModel(nn.Module):
    def __init__(self, modelTeacher, modelStudent):
        super(EnsembleTSModel, self).__init__()

        if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
            modelTeacher = modelTeacher.module
        if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
            modelStudent = modelStudent.module

        self.modelTeacher = modelTeacher
        self.modelStudent = modelStudent


class EnsembleClsErModel(nn.Module):
    def __init__(self, modelWorking, modelStable, modelPlastic):
        super(EnsembleClsErModel, self).__init__()

        if isinstance(modelWorking, (DistributedDataParallel, DataParallel)):
            modelWorking = modelWorking.module
        if isinstance(modelStable, (DistributedDataParallel, DataParallel)):
            modelStable = modelStable.module
        if isinstance(modelPlastic, (DistributedDataParallel, DataParallel)):
            modelPlastic = modelPlastic.module

        self.modelWorking = modelWorking
        self.modelStable = modelStable
        self.modelPlastic = modelPlastic