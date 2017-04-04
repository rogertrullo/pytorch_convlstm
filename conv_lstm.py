import torch.nn as nn
from torch.autograd import Variable
import torch

class Conv_LSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
      
    """
    def __init__(self, shape, input_chans, filter_size, num_features, batch_size):
        super(Conv_LSTM, self).__init__()
        
        self.shape = shape#H,W
        self.input_chans=input_chans
        self.filter_size=filter_size
        self.num_features = num_features
        self.batch_size=batch_size
        self.padding=(filter_size-1)/2#in this way the output has the same size
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4*self.num_features, self.filter_size, 1, self.padding)

    
    def forward(self, input, hidden_state):
        hidden,c=hidden_state#hidden and c are images with several channels
        print 'hidden ',hidden.size()
        print 'input ',input.size()
        combined = torch.cat((input, hidden), 1)#oncatenate in the channels
        A=self.conv(combined)
        (ai,af,ao,ag)=torch.split(A,self.num_features,dim=1)#it should return 4 tensors
        i=torch.sigmoid(ai)
        f=torch.sigmoid(af)
        o=torch.sigmoid(ao)
        g=torch.tanh(ag)
        
        next_c=f*c+i*g
        next_h=o*torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self):
        return (Variable(torch.zeros(self.batch_size,self.num_features,self.shape[0],self.shape[1])),Variable(torch.zeros(self.batch_size,self.num_features,self.shape[0],self.shape[1])))
    
    
num_features=10
filter_size=5
batch_size=10
shape=(25,25)#H,W
inp_chans=3

input = Variable(torch.rand(batch_size,inp_chans,shape[0],shape[1]))
conv_lstm=Conv_LSTM(shape, inp_chans, filter_size, num_features, batch_size)
hidden_state=conv_lstm.init_hidden()
print 'hidden_h shape ',hidden_state[0].size()
next_h, next_c=conv_lstm(input, hidden_state)
print next_h

#possible usage:
seq_len=10#time steps
x=Variable(torch.rand(batch_size,seq_len,inp_chans,shape[0],shape[1]))
loss=0
for t in range(seq_len):
    hidden_state=conv_lstm(x[:,t], hidden_state)
    #loss+=my_loss(hidden_state[0],target[t])#here we would use a loss
    print hidden_state[0].size()
loss.backward()