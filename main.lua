--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

Trainer = require 'train'

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Data loading
print('Creating Data Loader')
local trainLoader, valLoader = DataLoader.create(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- The trainer handles the training loop and evaluation on validation set
print('Creating Trainer')
local trainer = Trainer(model, criterion, opt, optimState)


local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestmAP = 0
for epoch = startEpoch, opt.nEpochs do
	-- Train for a single epoch
	local trainTop1, trainLoss = trainer:train(opt, epoch, trainLoader)

	-- Evaluate on the evaluation dataset
	local testTop1, evalLoss = trainer:test_top1(opt, epoch, valLoader)

	local bestModel = false
	if testTop1 < bestTop1 then
		bestModel = true
		bestTop1 = testTop1
		-- bestmAP = mAP 
		-- print(' * Best model ', testTop1, mAP)
	end

   -- local score = {trainTop1, testTop1, mAP}
   -- checkpoints.save(epoch, model, trainer.optimState, bestModel, opt, score)
end