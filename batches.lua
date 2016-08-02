require 'torch';
require 'nn';
require 'hdf5';
require 'lfs';
require 'os';
require 'string';

tasks = {'train','test','validation'}
batch_types = {'batch_input','batch_input_nnx','batch_target','batch_tobe_excluded_outcomes', 'batch_mu', 'batch_std'}
batch_size = 256
n_features = 30
n_time = 12
n_outcomes = 1
-- Ignores the remaining training examples after dividing by the batch size

-- Remove any files in the batch directory

for i = 1, #tasks do
	task = tasks[i]
	dir = batches_dir..task..'/'
	for file in lfs.dir(dir) do
		if(string.match(file, 'bix') ~= nil) then
			os.remove(dir..file)
		end
	end
end

-- Read data

batches = {}
fin = hdf5.open(in_fname, 'r')
for i = 1, #tasks do
	task = tasks[i]
	batches[task] = {} 
	for j = 1, #batch_types do
		batch_type = batch_types[j]
		if (batch_type == 'batch_input') then
			batches[task][batch_type] = fin:read('/'..batch_type..'_'..task):all()
			n_examples = batches[task][batch_type]:size()[1]
		elseif (batch_type == 'batch_input_nnx') then	
			batches[task][batch_type] = fin:read('/'..batch_type..'_'..task):all()
		elseif (batch_type == 'batch_target') then
			batches[task][batch_type] = fin:read('/'..batch_type..'_'..task):all()
		elseif (batch_type == 'batch_tobe_excluded_outcomes') then
			--[[
			fin2 = hdf5.open(tobe_excluded_fname)
			batches[task][batch_type] = fin2:read('/batch_tobe_excluded_outcomes_'..task):all()
			fin2:close()
			--]]
			batches[task][batch_type] = torch.zeros(n_examples, n_outcomes, 1, 1)
		elseif (batch_type == 'batch_mu') then
			batches[task][batch_type] = torch.zeros(n_examples, 1, n_features, n_time)	
		elseif (batch_type == 'batch_std') then
			batches[task][batch_type] = torch.ones(n_examples, 1, n_features, n_time)	
		end
	end
end
fin:close()

-- Create batches 

for i = 1, #tasks do
	task = tasks[i]
	n_examples = batches[task][batch_types[1]]:size()[1]
	n_batches = math.floor(n_examples / batch_size)
	perm_indices = torch.randperm(n_examples)

	for j = 1, #batch_types do
		batch_type = batch_types[j]
		for k = 1, n_batches do
			start_index = (k-1)*batch_size + 1
  		end_index = k*batch_size
			
			B = torch.Tensor(batch_size, batches[task][batch_type]:size()[2], batches[task][batch_type]:size()[3], batches[task][batch_type]:size()[4])
			li = 1
			for l = start_index, end_index do
				lp = perm_indices[l]
				B[{{li},{},{},{}}] = batches[task][batch_type][{{lp},{},{},{}}] 
				li = li + 1
			end

			torch.save(batches_dir..task..'/bix'..k..'_'..batch_type, B)
		end

		B = torch.Tensor(batch_size, batches[task][batch_type]:size()[2], batches[task][batch_type]:size()[3], batches[task][batch_type]:size()[4])
		li = 1
		start_index = n_batches*batch_size
		end_index = n_examples
		for l = start_index, end_index do
			lp = perm_indices[l]
			B[{{li},{},{},{}}] = batches[task][batch_type][{{lp},{},{},{}}] 
			li = li + 1
		end

		start_index = 1
		end_index = batch_size - (end_index - start_index)
		for l = start_index, end_index do
			lp = perm_indices[l]
			B[{{li},{},{},{}}] = batches[task][batch_type][{{lp},{},{},{}}] 
			li = li + 1
		end

		k2 = n_batches + 1
		torch.save(batches_dir..task..'/bix'..k2..'_'..batch_type, B)
	
	end
end



