from siamese.finetune import train as siamese_train
from cross.finetune import train as cross_train

print("===========CrossFineTuneModel===========")
cross_train()

print("==========SiameseFineTuneModel==========")
siamese_train()
