require 'cunn';
require 'torch';
require 'nn';
require 'hdf5';

local opt = lapp[[
	--inPath (default ''),
	--outDir (default ''), 
	--modelPath (default '')
	--n_labs (default 15)
	--n_time (default 12)
	--use_split_data (default 1)
	--modelDefn (default 1)
]]

inPath = opt.inPath
modelPath = opt.modelPath
outDir = opt.outDir
n_labs = opt.n_labs
n_time = opt.n_time
use_split_data = opt.use_split_data
modelDefn = opt.modelDefn

fullModel = torch.load(modelPath)

if (modelDefn == 1) then
	model = nn.Sequential()
	model:add(fullModel.modules[1].modules[1])
	model:add(fullModel.modules[1].modules[2])
	model:add(fullModel.modules[1].modules[3])
elseif (modelDefn == 2) then
	model = fullModel
elseif (modelDefn == 3) then
	model = fullModel
else 
	model = fullModel.modules[1]
end 

X = {}

inFile = hdf5.open(inPath, 'r')
if (use_split_data == 1) then
	X['train'] = inFile:read('/batch_input_train')
	X['validation'] = inFile:read('/batch_input_validation') 
	X['test'] = inFile:read('/batch_input_test')
	datasets = {'train','validation','test'}
	batch_size = 1000
else
	X['emb_features'] = inFile:read('/X_scaled')
	datasets = {'emb_features'}
	batch_size = 10000
end

for d = 1, #datasets do 

	dataset = datasets[d]

	print(dataset)

	n_batches = math.floor(X[dataset]:dataspaceSize()[1] / batch_size)

	for i = 1, n_batches do
		print(i..'/'..n_batches)

		start = (i - 1) * batch_size + 1
		if (i == n_batches) then
			stop = X[dataset]:dataspaceSize()[1]
		else
			stop = i*batch_size
		end
		print(start)
		print(stop)
		n_examples = stop - start + 1

		if (modelDefn == 1) or (modelDefn == 2) then
			X_batch = X[dataset]:partial({start, stop}, {1, 1}, {1, n_labs}, {1, X[dataset]:dataspaceSize()[4]})
			X_batch_padded = torch.zeros(n_examples, 1, n_labs, n_time)
			for j = 1, X_batch:size()[1] do
				for k = 1, X_batch:size()[3] do
					for l = 1, X_batch:size()[4] do
						X_batch_padded[{j,1,k,l}] = X_batch[{j,1,k,l}]
					end
				end
			end
		else
			X_batch_padded = X[dataset]:partial({start, stop}, {1, 1}, {1, n_labs}, {1, n_time})
		end

		outFile = hdf5.open(outDir..dataset..'_batch_'..i..'.h5', 'w')

		if (modelDefn == 1) or (modelDefn == 4) then
			R = model:forward(X_batch_padded:cuda()):clone()
			outFile:write('/X_scaled', R:double())
		else
			R = model:forward(X_batch_padded:cuda())
			Rout = torch.zeros(R[1]:size()[1], #R)
			for j = 1,R[1]:size()[1] do
				for k = 1,#R do
					p_0 = R[k][{j,1}]
					p_1 = R[k][{j,2}]
					Rout[{j,k}] = p_1
					--if (p_1 >= p_0) then
					--	Rout[{j,k}] = 1
					--end
				end
			end
			outFile:write('/X_scaled', Rout:double())
		end
		outFile:close()

	end
 
end

inFile:close()

