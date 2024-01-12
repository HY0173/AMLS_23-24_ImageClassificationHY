import torch
import torch.nn as nn
import medmnist
from medmnist.evaluator import Evaluator

def test_B(model,device,split,data_loader):
    testLoss=0
    model.eval()
    y_true = torch.tensor([])       # the ground truth labels
    y_score = torch.tensor([])      # the predicted score of each class
    
    # To device
    y_true = y_true.to(device)
    y_score = y_score.to(device)

    evaluator = Evaluator('pathmnist', split)
    if split == 'val':
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                targets = targets.squeeze().long()
                batch_loss = criterion(outputs, targets)
                testLoss += batch_loss.item()/len(data_loader)

                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)
            
            y_true = y_true.cpu().numpy()
            y_score = y_score.detach().cpu().numpy()
            
            metrics = evaluator.evaluate(y_score)
    
            print('%s  auc: %.3f  acc:%.3f' % ('val', *metrics))
            return testLoss

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            targets = targets.squeeze().long()

            outputs = outputs.softmax(dim=-1)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        
        metrics = evaluator.evaluate(y_score)
    
        print('%s  auc: %.3f  acc:%.3f' % ('test', *metrics))
