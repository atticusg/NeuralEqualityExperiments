import torch
from torch.nn import functional as F

class ReLULSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, dropout):
        """"Constructor of the class"""
        super(ReLULSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(p=dropout)
        self.batch_first = batch_first

        ih, hh = [], []
        for i in range(num_layers):
            ih.append(torch.nn.Linear(input_size, 4 * hidden_size))
            hh.append(torch.nn.Linear(hidden_size, 4 * hidden_size))
        self.w_ih = torch.nn.ModuleList(ih)
        self.w_hh = torch.nn.ModuleList(hh)

    def forward(self, input, hidden):
        #Define hidden states and change dimension order if batch_first is set
        if self.batch_first:
            batch_size = input.size()[0]
        else:
            batch_size = input.size()[1]
        if hidden is None:
            hidden = (torch.zeros((self.num_layers,batch_size,self.hidden_size)),torch.zeros((self.num_layers,batch_size ,self.hidden_size)))
        elif self.batch_first:
            hidden = (hidden[0].transpose(0,1),hidden[1].transpose(0,1))
        if self.batch_first:
            input = input.transpose(0,1)
        output = []
        nhx, ncx = hidden[0][0].unsqueeze(0), hidden[1][0].unsqueeze(0)
        for i in range(input.size()[0]):
            layer_input = input[i]
            hy, cy = [], []
            for i in range(self.num_layers):
                hx, cx = nhx, ncx
                gates = self.w_ih[i](layer_input) + self.w_hh[i](hx)
                i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 2)
                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                c_gate = F.relu(c_gate)
                o_gate = torch.sigmoid(o_gate)
                ncx = (f_gate * cx) + (i_gate * c_gate)
                nhx = o_gate * F.relu(ncx)
                layer_input = nhx
                cy.append(ncx)
                hy.append(nhx)
                layer_input = self.dropout(nhx)
            hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)
            output.append(hy[-1].squeeze(0))
        output = torch.stack(output, 0)
        if self.batch_first:
            hy = hy.transpose(1,2)
            cy = cy.transpose(1,2)
            output = output.transpose(0,1)
            hy,cy = hy.squeeze(0).squeeze(1).unsqueeze(1), cy.squeeze(0).squeeze(1).unsqueeze(1)
        else:
            hy,cy = hy.squeeze(0).squeeze(1).unsqueeze(0), cy.squeeze(0).squeeze(1).unsqueeze(0)
        return output, (hy,cy)
