from tqdm import tqdm
from . import Test_A
from Test_A import test_A
import torch
import numpy as np

import matplotlib.pyplot as plt

def Train_A(model,train_loader,val_loader,test_loader,NUM_EPOCHS,criterion,optimizer,device):
    model = model.to(device)
    PlotTrainLoss=[]
    PlotValLoss=[]
    BATCH_SIZE = len(train_loader)

    # For each epoch
    for epoch in range(NUM_EPOCHS):
        avg_loss = 0
        train_correct = 0
        all_count = 0

        # Train mode  -->  forward + backward + optimize
        print("-----------------------Epoch{}-----------------------".format(epoch+1))
        model.train()
        for inputs, targets in tqdm(train_loader):
            # Data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Set parameter gradients to zero
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            targets = targets.squeeze().long()
            # Forward-pass (criterion: loss function, such as CrossEntropyLoss)
            loss = criterion(outputs, targets)

            # Backward-pass
            loss.backward()

            # Update weights
            optimizer.step()

            avg_loss += loss.item()

            # Compute ACC of Training_set
            outputs = torch.argmax(outputs.softmax(dim=-1),dim=1)
            all_count += len(targets)
            correct = len(targets)-torch.count_nonzero(outputs-targets)
            train_correct += correct

        # Compute ACC of Training_set
        acc_train = train_correct/all_count
        print("Training Accuracy: ",acc_train)
        # Compute Loss (sum(loss)/batch_size)
        avg_loss = avg_loss/BATCH_SIZE
        # Test on validation_set
        avg_loss_val = test_A(model,device,'val',val_loader)

        PlotTrainLoss.append(avg_loss)
        PlotValLoss.append(avg_loss_val)

    test_A(model,device,'test',test_loader)

    return PlotTrainLoss,PlotValLoss

# Plot
def plot_loss(NUM_EPOCHS,PlotTrainLoss,PlotValLoss):
    plt.plot(range(1,NUM_EPOCHS+1),PlotTrainLoss,label='TrainLoss')
    plt.plot(range(1,NUM_EPOCHS+1), PlotValLoss, label='ValLoss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

