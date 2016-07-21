require 'cunn';
require 'torch';
require 'nn';
require 'hdf5';

local opt = lapp[[
	--inPath (default ''),
	--outDir (default ''), 
	--modelPath (default '')
	--n_labs (default 18)
	--n_time (default 36)
]]

inPath = opt.inPath
modelPath = opt.modelPath
outDir = opt.outDir
n_labs = opt.n_labs
n_time = opt.n_time
datasets = {'emb_features'}--{'train','validation','test'}
batch_size = 10000

fullModel = torch.load(modelPath)
model = nn.Sequential()
model:add(fullModel.modules[1].modules[1])
model:add(fullModel.modules[1].modules[2])
model:add(fullModel.modules[1].modules[3])

X = {}

inFile = hdf5.open(inPath, 'r')
X['emb_features'] = inFile:read('/X_scaled')
--X['train'] = inFile:read('/batch_input_train')
--X['validation'] = inFile:read('/batch_input_validation') 
--X['test'] = inFile:read('/batch_input_test')

for d = 1, #datasets do 

	dataset = datasets[d]

	print(dataset)

	n_batches = math.floor(X[dataset]:dataspaceSize()[1] / batch_size)

	for i = 1, n_batches do
		print(i..'/'..n_batches)

		start = i * batch_size
		if (i == n_batches) then
			stop = X[dataset]:dataspaceSize()[1]
		else
			stop = (i+1)*batch_size
		end
		n_examples = stop - start + 1
		X_batch = X[dataset]:partial({start, stop}, {1, 1}, {1, n_labs}, {1, X[dataset]:dataspaceSize()[4]})
		X_batch_padded = torch.zeros(n_examples, 1, n_labs, n_time)
		for j = 1, X_batch:size()[1] do
			for k = 1, X_batch:size()[3] do
				for l = 1, X_batch:size()[4] do
					X_batch_padded[{j,1,k,l}] = X_batch[{j,1,k,l}]
				end
			end
		end
		R = model:forward(X_batch_padded:cuda()):clone()
		outFile = hdf5.open(outDir..dataset..'_batch_'..i..'.h5', 'w')
		outFile:write('/X_scaled', R:double())
		outFile:close()
	end
 
end

inFile:close()

