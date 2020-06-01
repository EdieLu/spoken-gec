import torch
import torch.nn as nn

# REF: https://github.com/eladhoffer/seq2seq.pytorch + with attention removed
# customised LSTM to allow reproduction
# currently broken

import warnings
warnings.filterwarnings("ignore")


class StackedCell(nn.Module):
	"""
		run through stacked cell layers (no recurrency)
		used only in TimeRecurrentCell

		inputs: input for layer0
		hidden: initial hidden state for all layers
		output: res
	"""
	def __init__(self, input_size, hidden_size, num_layers=1,
				 dropout=0, bias=True, rnn_cell=nn.LSTMCell, residual=False):
		super(StackedCell, self).__init__()

		self.dropout = nn.Dropout(dropout)
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.residual = residual
		self.layers = nn.ModuleList()
		for _ in range(num_layers):
			rnn = rnn_cell(input_size, hidden_size, bias=bias)
			self.layers.append(rnn)
			input_size = hidden_size

	def forward(self, inputs, hidden):
		def select_layer(h_state, i):
			if isinstance(h_state, tuple):
				return tuple([select_layer(s, i) for s in h_state])
			else:
				return h_state[i]

		next_hidden = []
		for i, layer in enumerate(self.layers):
			next_hidden_i = layer(inputs, select_layer(hidden, i))
			output = next_hidden_i[0] if isinstance(next_hidden_i, tuple) \
				else next_hidden_i
			if i + 1 < self.num_layers:
				output = self.dropout(output)
			if self.residual and inputs.size(-1) == output.size(-1):
				inputs = output + inputs
			else:
				inputs = output
			next_hidden.append(next_hidden_i)
		if isinstance(hidden, tuple):
			next_hidden = tuple([torch.stack(h) for h in zip(*next_hidden)])
		else:
			next_hidden = torch.stack(next_hidden)
		return inputs, next_hidden


class TimeRecurrentCell(nn.Module):
	"""
		customise recurrent layer using cell (manually loop over time)
		allow time reversal while keeping reproducibility
		LSTMcell: (h_1, c_1) =  LSTMcell(input, (h_0, c_0))

		inputs: b x t x input_size
		outputs: b x t x cell_hidden_size
		hidden: b x t x cell_hidden_size

		note: 	in pytorch default is batch_first=False
				[seq_len, batch_size, var_dim]
	"""
	def __init__(self, cell, batch_first=True, lstm=True, reverse=False):

		super(TimeRecurrentCell, self).__init__()
		self.cell = cell
		self.lstm = lstm
		self.reverse = reverse
		self.batch_first = batch_first

	def forward(self, inputs, hidden=None):

		hidden_size = self.cell.hidden_size
		batch_dim = 0 if self.batch_first else 1
		time_dim = 1 if self.batch_first else 0
		batch_size = inputs.size(batch_dim)

		# init hidden:
		# 	a. set as 0: hidden[0] - num_layers x b x hidden_size
		# 	b. allow learning:
		if hidden is None:
			num_layers = getattr(self.cell, 'num_layers', 1)
			zero = inputs.data.new(1).zero_()
			h0 = zero.view(1, 1, 1).expand(num_layers, batch_size, hidden_size)
			hidden = h0
			if self.lstm:
				hidden = (hidden, h0)

		# traverse through time
		outputs = []
		inputs_time = list(inputs.split(1, time_dim)) # [b x 1 x var_dim] * t chunks
		# print('[TimeRecurrentCell]')
		# print('time steps: {}'.format(len(inputs_time)))

		if self.reverse:
			inputs_time.reverse()
		for input_t in inputs_time:
			input_t = input_t.squeeze(time_dim) # b x var_dim
			output_t, hidden = self.cell(input_t, hidden) # output_t: b x hidden_size
			outputs += [output_t]
		if self.reverse:
			outputs.reverse()
			# print('here')
		outputs = torch.stack(outputs, time_dim) # b x t x hidden_size
		# print('outputs size: {}'.format(outputs.size()))

		return outputs, hidden


class ConcatRecurrent(nn.Sequential):
	"""
		concat output of rnn layers
		inputs: common input for all layers
		hidden: common initial hidden state for all layers
		output: concat[res1, res2, res3, ...]
	"""
	def forward(self, inputs, hidden=None):
		hidden = hidden or tuple([None] * len(self))
		next_hidden = []
		outputs = []
		print(self._modules)
		for i, module in enumerate(self._modules.values()):
			print(i, module)
			curr_output, h = module(inputs, hidden[i])
			outputs.append(curr_output)
			next_hidden.append(h)
		output = torch.cat(outputs, -1)
		return output, tuple(next_hidden)


class StackedRecurrent(nn.Sequential):
	"""
		run through stacked complex-rnn layers
		inputs: input for layer0
		hidden: initial hidden state for layer0
		output: res
	"""
	def __init__(self, dropout=0, residual=False):
		super(StackedRecurrent, self).__init__()
		self.residual = residual
		self.dropout = dropout

	def forward(self, inputs, hidden=None):
		hidden = hidden or tuple([None] * len(self))
		next_hidden = []
		for i, module in enumerate(self._modules.values()):
			output, h = module(inputs, hidden[i])
			next_hidden.append(h)
			if self.residual and inputs.size(-1) == output.size(-1):
				inputs = output + inputs
			else:
				inputs = output
			inputs = nn.functional.dropout(
				inputs, self.dropout, self.training)

		return output, tuple(next_hidden)


def CustomiseLSTM(input_size, hidden_size,
			num_layers=1, batch_first=True,
			dropout=0, bidirectional=False, residual=False):
	"""
		customise LSTM: default biLSTM not reproducible
		hierarchy: cell -> StackedCell -> TimeRecurrentCell -> Concat/StackedRecurrent

		E.g.
		multi-layer unilstm
			a. along layer: in[t=0], h[l=0] -> h[l=1] -> h[l=L]
			b. along time: 	in[t=0], h[l=:L] -> in[t=1], h[l=:L] -> in[t=T], h[l=:L]
		multi-layer bilstm
			opt a: stack multilayer forward + backward unilstm
				(does not allow forward/backward talk between layers)
			opt b: use previous bilstm hidden state as the input of the next bilstm layer
	"""

	if bidirectional:

		# opt a
		# cell = StackedCell(rnn_cell=nn.LSTMCell,
		# 				   input_size=input_size,
		# 				   hidden_size=hidden_size,
		# 				   num_layers=num_layers,
		# 				   residual=residual,
		# 				   dropout=dropout)

		cell = nn.LSTMCell
		bi_module = ConcatRecurrent()
		bi_module.add_module('0_forward', TimeRecurrentCell(cell(input_size, hidden_size),
			batch_first=batch_first, lstm=True))
		bi_module.add_module('0_reversed', TimeRecurrentCell(cell(input_size, hidden_size),
	 		batch_first=batch_first, lstm=True, reverse=True))
		module = StackedRecurrent(residual)
		# for i in range(num_layers):
		for i in range(2):
			module.add_module(str(i), bi_module)
		# module.add_module(str(0), bi_module)

		# for name, param in bi_module.named_parameters():
		# 	print(name, param.size())


		"""
		# opt b
		cell = StackedCell(rnn_cell=nn.LSTMCell,
		   input_size=input_size,
		   hidden_size=hidden_size,
		   num_layers=1,
		   residual=residual,
		   dropout=dropout)

		bi_module = ConcatRecurrent()
		bi_module.add_module('0', TimeRecurrentCell(cell,
			batch_first=batch_first,
			lstm=True))
		bi_module.add_module('0_reversed', TimeRecurrentCell(cell,
			 batch_first=batch_first,
			 lstm=True,
			 reverse=True))
		module = StackedRecurrent(residual)
		for i in range(num_layers):
			module.add_module(str(i), bi_module)
		"""

	else:
		cell = StackedCell(rnn_cell=nn.LSTMCell,
						   input_size=input_size,
						   hidden_size=hidden_size,
						   num_layers=num_layers,
						   residual=residual,
						   dropout=dropout)

		module = TimeRecurrentCell(cell,
								   batch_first=batch_first,
								   lstm=True)

	return module
