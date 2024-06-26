import torch 
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def train(model, use_cuda, train_loader, optimizer, tr_loss, epoch, log_interval):
    """
    Train one epoch
    
    model -- the neural network
    use_cuda -- true if GPU should be used
    train_loader -- data loader
    optimizer -- network optimizer
    tr_loss -- list to save the loss
    epoch -- number of current epoch
    log_interval -- number of training steps between logs
    """
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # for each batch
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
    
        optimizer.zero_grad()
        
        output = model(data)

        loss = model.loss_function(output, data)
        
        loss.backward()
        
        optimizer.step()

        curr_loss = loss.item()
        tr_loss.append(curr_loss)
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), curr_loss))


def validate(model, use_cuda, test_loader, test_loss, plot):
    """
    Compute test metrics
    
    model -- the neural network
    use_cuda -- true if GPU should be used
    test_loader -- data loader
    plot -- bool if results are plotted
    """
    # set model to evaluation mode
    model.eval()
    correct = 0
    plotted = 0
    test_loss_tmp = 0
    
    # disable gradients globally
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # for each batch
            if use_cuda:
                # transfer to GPU
                data = data.cuda()
                target = target.cuda()
            
            # run network and compute metrics
            output = model(data)
            test_loss_tmp += model.loss_function(output, data).item() # sum up batch loss
            
    test_loss_tmp /= len(test_loader.dataset)
    test_loss.append(test_loss_tmp)

    # show results
    print('\nTest set: Average loss: {:.8f}\n'.format(
        test_loss_tmp))
    
    if use_cuda: 
        output = output[0].cpu()
        data = data.cpu()
        target = target.cpu()
    else: 
        output = output[0]
    
    if plot: 
        fig, axes = plt.subplots(5, 10, figsize=(15, 7))
        for i in range(5):
            tmp_idx = 0
            for j in range(5):
                # Plot original images
                axes[i, j + tmp_idx].imshow(data[i * 5 + j].squeeze().numpy(), cmap='gray')
                axes[i, j+ tmp_idx].axis('off')
                axes[i, j+ tmp_idx].set_title(f'Orig: {target[i * 5 + j].item()}')
                # Plot reconstructed images
                tmp_idx += 1
                axes[i, j + tmp_idx].imshow(output[i * 5 + j].squeeze().detach().numpy(), cmap='gray')
                axes[i, j + tmp_idx].axis('off')
                axes[i, j + tmp_idx].set_title(f'Recon: {target[i * 5 + j].item()}')
        plt.show()


def plot_training(tr_loss_step, tr_loss, test_loss, epochs, train_loader, batch_size):
    ''' 
    Plots the losses of the training process.
    --------------------------
    tr_loss_step: training loss per step as a List
    tr_loss: average training loss as a list
    test_loss: average training loss as a list
    epochs: the total amount of epochst trained (to syncronize the averages with the steps)
    train_loader: ""
    batch_size: ""
    '''
    smooth_tr_loss = savgol_filter(tr_loss_step, window_length=100, polyorder=2)


    # Erstellen des Plots
    plt.figure(figsize=(10, 6))

    # Plotten der originalen Kurven
    steps_per_epoch = np.arange(int(len(train_loader.dataset)/batch_size), int(epochs * len(train_loader.dataset)/batch_size), 
                            int(len(train_loader.dataset)/batch_size))

    plt.plot(tr_loss_step, label='Training Loss', color='green', linestyle='-', marker='')
    plt.plot(steps_per_epoch, tr_loss, label='Training Average Loss', color='blue', linestyle='-', marker='o')
    plt.plot(steps_per_epoch, test_loss, label='Validation Average Loss', color='red', linestyle='-', marker='o')


    # Plotten der geglätteten Kurven
    plt.plot(smooth_tr_loss, label='Smoothed Training Loss', color='cyan', linestyle='--')

    # Hinzufügen von Legende und Labels
    plt.legend(loc='best')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.yscale('log')
    # Anzeigen des Plots
    plt.show()