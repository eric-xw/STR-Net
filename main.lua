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
-- local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

Trainer = require 'train'

-- Load previous checkpoint, if it exists
-- local checkpoint, optimState = checkpoints.latest(opt)

-- Data loading
print('Creating Data Loader')
local trainLoader, valLoader = DataLoader.create(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- The trainer handles the training loop and evaluation on validation set
print('Creating Trainer')
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   --local top1Err, top5Err = trainer:test(opt, 0, valLoader)
   --print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))

   local AP = trainer:test2(opt, 0, val2Loader)
   local mAP = AP:mean()
   print(string.format(' * Results mAP: %6.3f', mAP))

   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestmAP = 0
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss = trainer:train(opt, epoch, trainLoader)
end