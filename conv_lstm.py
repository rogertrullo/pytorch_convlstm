import torch.nn as nn
from torch.autograd import Variable
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
      
    """
    def __init__(self, shape, input_chans, filter_size, num_features):
        super(CLSTM_cell, self).__init__()
        
        self.shape = shape#H,W
        self.input_chans=input_chans
        self.filter_size=filter_size
        self.num_features = num_features
        #self.batch_size=batch_size
        self.padding=(filter_size-1)/2#in this way the output has the same size
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4*self.num_features, self.filter_size, 1, self.padding)

    
    def forward(self, input, hidden_state):
        hidden,c=hidden_state#hidden and c are images with several channels
        #print 'hidden ',hidden.size()
        #print 'input ',input.size()
        combined = torch.cat((input, hidden), 1)#oncatenate in the channels
        #print 'combined',combined.size()
        A=self.conv(combined)
        (ai,af,ao,ag)=torch.split(A,self.num_features,dim=1)#it should return 4 tensors
        i=torch.sigmoid(ai)
        f=torch.sigmoid(af)
        o=torch.sigmoid(ao)
        g=torch.tanh(ag)
        
        next_c=f*c+i*g
        next_h=o*torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self,batch_size):
        return (Variable(torch.zeros(batch_size,self.num_features,self.shape[0],self.shape[1])).cuda(),Variable(torch.zeros(batch_size,self.num_features,self.shape[0],self.shape[1])).cuda())


class CLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
      
    """
    def __init__(self, shape, input_chans, filter_size, num_features,num_layers):
        super(CLSTM, self).__init__()
        
        self.shape = shape#H,W
        self.input_chans=input_chans
        self.filter_size=filter_size
        self.num_features = num_features
        self.num_layers=num_layers
        cell_list=[]
        cell_list.append(CLSTM_cell(self.shape, self.input_chans, self.filter_size, self.num_features).cuda())#the first
        #one has a different number of input channels
        
        for idcell in xrange(1,self.num_layers):
            cell_list.append(CLSTM_cell(self.shape, self.num_features, self.filter_size, self.num_features).cuda())
        self.cell_list=nn.ModuleList(cell_list)      

    
    def forward(self, input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W

        """

        current_input = input.transpose(0, 1)#now is seq_len,B,C,H,W
        #current_input=input
        next_hidden=[]#hidden states(h and c)
        seq_len=current_input.size(0)

        
        for idlayer in xrange(self.num_layers):#loop for every layer

            hidden_c=hidden_state[idlayer]#hidden and c are images with several channels
            all_output = []
            output_inner = []            
            for t in xrange(seq_len):#loop for every step
                hidden_c=self.cell_list[idlayer](current_input[t,...],hidden_c)#cell_list is a list with different conv_lstms 1 for every layer

                output_inner.append(hidden_c[0])

            next_hidden.append(hidden_c)
            current_input = torch.cat(output_inner, 0).view(current_input.size(0), *output_inner[0].size())#seq_len,B,chans,H,W


        return next_hidden, current_input

    def init_hidden(self,batch_size):
        init_states=[]#this is a list of tuples
        for i in xrange(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states
###########Usage#######################################    
num_features=10
filter_size=5
batch_size=10
shape=(25,25)#H,W
inp_chans=3
nlayers=2
seq_len=4

#If using this format, then we need to transpose in CLSTM
input = Variable(torch.rand(batch_size,seq_len,inp_chans,shape[0],shape[1])).cuda()

conv_lstm=CLSTM(shape, inp_chans, filter_size, num_features,nlayers)
conv_lstm.apply(weights_init)
conv_lstm.cuda()

print 'convlstm module:',conv_lstm


print 'params:'
params=conv_lstm.parameters()
for p in params:
   print 'param ',p.size()
   print 'mean ',torch.mean(p)


hidden_state=conv_lstm.init_hidden(batch_size)
print 'hidden_h shape ',len(hidden_state)
print 'hidden_h shape ',hidden_state[0][0].size()
out=conv_lstm(input,hidden_state)
print 'out shape',out[1].size()
print 'len hidden ', len(out[0])
print 'next hidden',out[0][0][0].size()
print 'convlstm dict',conv_lstm.state_dict().keys()


L=torch.sum(out[1])
L.backward()
