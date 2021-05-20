import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
import datetime

from netClasses import *
from netFunctions import * 
from plotFunctions import * 

parser = argparse.ArgumentParser(description='Updates of Equilibrium Prop Match Gradients of Backprop Through Time in an RNN with Static Input')
parser.add_argument(
    '--batch-size',
    type=int,
    default=600,
    metavar='N',
    help='input batch size for training (default: 20)')
parser.add_argument(
    '--update-batch-size',
    type=int,
    default=20,
    metavar='N',
    help='subsampling batch size for training (default: 10)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    metavar='N',
    help='input batch size for testing (default: 1000)')   
parser.add_argument(
    '--epochs',
    type=int,
    default=500,
    metavar='N',
help='number of epochs to train (default: 1)')    
parser.add_argument(
    '--lr_tab',
    nargs = '+',
    type=float,
    default=[0.025, 0.028, 0.4], # for the best
    metavar='LR',
    help='learning rate (default: [0.05, 0.1])')
parser.add_argument(
    '--size_tab',
    nargs = '+',
    type=int,
    default=[10],
    metavar='ST',
    help='tab of layer sizes (default: [10])')      
parser.add_argument(
    '--dt',
    type=float,
    default=0.2,
    metavar='DT',
    help='time discretization (default: 0.2)') 
parser.add_argument(
    '--T',
    type=int,
    default=130,
    metavar='T',
    help='number of time steps in the forward pass (default: 100)')
parser.add_argument(
    '--Kmax',
    type=int,
    default=30,
    metavar='Kmax',
    help='number of time steps in the backward pass (default: 25)')  
parser.add_argument(
    '--beta',
    type=float,
    default=0.01,
    metavar='BETA',
    help='nudging parameter (default: 1)') 
parser.add_argument(
    '--pred_inps',
    nargs = '+',
    type=int,
    default=[4, 20, 80],
    metavar='Pred_inputs',
    help='prediction inputs (default: [])') 
parser.add_argument(
    '--delay',
    type=int,
    default=110,
    metavar='Delay',
    help='delay for the clamped (default: 110)') 
parser.add_argument(
    '--training-method',
    type=str,
    default='eqprop',
    metavar='TMETHOD',
    help='training method (default: eqprop)')
parser.add_argument(
    '--action',
    type=str,
    default='train',
    help='action to execute (default: train)')    
parser.add_argument(
    '--activation-function',
    type=str,
    default='hardsigm',
    metavar='ACTFUN',
    help='activation function (default: sigmoid)')
parser.add_argument(
    '--no-clamp',
    action='store_true',
    default=False,
    help='clamp neurons between 0 and 1 (default: True)')
parser.add_argument(
    '--discrete',
    action='store_true',
    default=False, 
    help='discrete-time dynamics (default: False)')
parser.add_argument(
    '--toymodel',
    action='store_true',
    default=False, 
    help='Implement fully connected toy model (default: False)')                                                    
parser.add_argument(
    '--device-label',
    type=int,
    default=0,
    help='selects cuda device (default 0, -1 to select )')
parser.add_argument(
    '--C_tab',
    nargs = '+',
    type=int,
    default=[512, 256, 3],
    metavar='LR',
    help='channel tab (default: [])')
parser.add_argument(
    '--padding',
    type=int,
    default=0,
    metavar='P',
    help='padding (default: 0)')
parser.add_argument(
    '--Fconv',
    type=int,
    default=3,
    metavar='F',
    help='convolution filter size (default: 5)')
parser.add_argument(
    '--Fpool',
    type=int,
    default=2,
    metavar='Fp',
    help='pooling filter size (default: 2)')         
parser.add_argument(
    '--benchmark',
    action='store_true',
    default=True, 
    help='benchmark EP wrt BPTT (default: False)')

args = parser.parse_args()

args.conv = not not args.C_tab

batch_size           = args.batch_size
update_batch_size    = args.update_batch_size
batch_size_test      = args.test_batch_size
pred_inps            = args.pred_inps
delay                = args.delay




class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)
        
        
class ReshapeTransformTarget:
    def __init__(self, number_classes):
        self.number_classes = number_classes
    
    def __call__(self, target):
        target=torch.tensor(target).unsqueeze(0).unsqueeze(1)
        target_onehot = torch.zeros((1,self.number_classes))      
        return target_onehot.scatter_(1, target, 1).squeeze(0)


if (args.conv):
    transforms=[torchvision.transforms.ToTensor()]
else:
    transforms=[torchvision.transforms.ToTensor(),ReshapeTransform((-1,))]


################################# these augmentations are based on https://github.com/Laborieux-Axel/Equilibrium-Propagation ##################################################

transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(0.5),
                                                  torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
                                                  torchvision.transforms.ToTensor(), 
                                                  torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                                   std=(3*0.2023, 3*0.1994, 3*0.2010)) ])   
transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                     torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                                      std=(3*0.2023, 3*0.1994, 3*0.2010)) ]) 

cifar10_train_dset = torchvision.datasets.CIFAR10('./cifar10_pytorch', train=True, transform=transform_train, download=True, target_transform=ReshapeTransformTarget(10))
cifar10_test_dset = torchvision.datasets.CIFAR10('./cifar10_pytorch', train=False, transform=transform_test, download=True , target_transform=ReshapeTransformTarget(10))

train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(cifar10_test_dset, batch_size=500, shuffle=False, num_workers=0)  

#################################################################################################################################################################################


if  args.activation_function == 'sigm':
    def rho(x):
        return 1/(1+torch.exp(-(4*(x-0.5))))
    def rhop(x):
        return 4*torch.mul(rho(x), 1 -rho(x))

elif args.activation_function == 'hardsigm':
    def rho(x):
        # this activation is based on https://github.com/Laborieux-Axel/Equilibrium-Propagation
        return (1+F.hardtanh(x-1))*0.5
    def rhop(x):
        return (x >= 0) & (x <= 1)

elif args.activation_function == 'tanh':
    def rho(x):
        return torch.tanh(x)
    def rhop(x):
        return 1 - torch.tanh(x)**2 
            
                    
if __name__ == '__main__':
  
    net = convEP(args)
      
    if args.action == 'train':

        #create path              
        BASE_PATH, name = createPath(args)

        #save hyperparameters
        createHyperparameterfile(BASE_PATH, name, args)
        
        #train with EP
        error_train_tab = []
        error_test_tab = []  

        #*****MEASURE ELAPSED TIME*****#
        start_time = datetime.datetime.now()
        #******************************#

        for epoch in range(1, args.epochs + 1):
            error_train = train(net, train_loader, epoch, batch_size, update_batch_size, pred_inps, delay, args.training_method)
            error_test = evaluate(net, test_loader)
            error_train_tab.append(error_train)
            error_test_tab.append(error_test) ;
            results_dict = {'error_train_tab' : error_train_tab, 'error_test_tab' : error_test_tab, 'elapsed_time': datetime.datetime.now() - start_time}  
            outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
            # save the parameters
            torch.save(net, BASE_PATH + '/model_epoch' + str(epoch) + '.pt')                       
            pickle.dump(results_dict, outfile)
            outfile.close()   


    elif args.action == 'receipe':
        receipe(net, train_loader, 20)
