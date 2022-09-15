import torch
from tqdm import tqdm
import numpy as np
import math
from src import pytorchtools

def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=100, device='cpu',patience=None):
    model = model.to(device)
    
    
    print('train(): model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))
    
    
    history = {}
    history['train_acc'] = []
    history['true_train_acc'] = []
    history['val_acc'] = []
    history['distance'] = []
    history['sq_loss'] = []
    history['loss'] = []
    history['loss_clean'] = []
    
        
    model.train()
    num_train_correct = 0
    num_train_correct_true = 0
    num_train_examples = 0
     
    num_val_correct  = 0
    num_val_examples = 0
    running_loss = 0.0
    pbar = tqdm(range(1, epochs+1))
    counter=0

    for batch in train_dl:
        counter+=1
        keep_track=False

        for _ in pbar:
            optimizer.zero_grad()
            x = batch[0].to(device)
            y = batch[1].to(device)
            yhat = model(x)
            print(yhat)
            print(y)
            y_true = batch[2].to(device)
            print(y_true)
            
            '''
            if len(batch)==3:
                keep_track=True
                y_true = batch[2].to(device)

                for i in range(len(x)):
                    if y[i]==y_true[i]:
                        loss = loss_fn(yhat[i], y[i])
                        loss_clean=loss.cpu().detach().numpy()
                        history['loss_clean'].append(loss_clean)
                    else:
                        loss = loss_fn(yhat[i], y[i])
                        loss_cor=loss.cpu().detach().numpy()
                        history['loss'].append(loss_cor)
            
                '''
            loss = loss_fn(yhat, y)
            
            loss.backward()
            optimizer.step()

            running_loss+=loss
        num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
        num_train_correct_true += (torch.max(yhat, 1)[1] == y_true).sum().item()
        num_train_examples += x.shape[0]


        train_acc = num_train_correct / num_train_examples
        true_train_acc = num_train_correct_true / num_train_examples
       
       
   
        for batch in val_dl:
            x = batch[0].to(device)
            y = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

        num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
        num_val_examples += y.shape[0]

        
        val_acc = num_val_correct / num_val_examples   
        history['val_acc'].append(val_acc)
        
        print("{}. batch, train acc: {}, true train acc: {}, test acc: {}".format(counter,train_acc,true_train_acc,val_acc))
        model.eval()
        

        history['train_acc'].append(train_acc)
        history['true_train_acc'].append(true_train_acc)
        history['sq_loss'].append(math.sqrt(running_loss))


    return history
