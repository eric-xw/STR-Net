--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--
--  Contributor: Xin Wang

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
    self.model = model
    self.criterion = criterion
    self.optimState = optimState or {
        -- originalLR = opt.LR,
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        nesterov = true,
        dampening = 0.0,
        weightDecay = opt.weightDecay,
    }
    self.opt = opt
    self.params, self.gradParams = model:getParameters()
    self.LR_decay_freq = opt.LR_decay_freq
end

function Trainer:train(opt, epoch, dataloader)
   -- Trains the model for a single epoch

    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local LRM = self:learningRateModifier(epoch) 
    self.optimState.learningRate = self.optimState.learningRate * LRM

    local function feval()
        return self.criterion.output, self.gradParams
    end

    local trainSize = dataloader:size()
    local top1Sum = 0.0
    local lossSum = 0.0
    local N = 0

    print('=> Training epoch # ' .. epoch)
    -- set the batch norm to training mode
    self.model:training()
    for n, sample in dataloader:run(true) do -- true: shuffle the data
        local dataTime = dataTimer:time().real

        -- Copy input and target to the GPU
        self:copyInputs(sample)
        local batchSize = self.input:size(1)
        local timesteps = self.input:size(2)
        self.input = self.input:view(batchSize * timesteps, -1)
        self.target = self.target:view(batchSize * timesteps, -1)

        local Q_o = torch.rand(batchSize * timesteps, opt.nObjects):cuda()
        local Q_v = torch.rand(batchSize * timesteps, opt.nVerbs):cuda()
        local inputTuple = {self.input, Q_o, Q_v}
        local output = self.model:forward(inputTuple):float()
        local loss = self.criterion:forward(self.model.output, self.target)

        self.model:zeroGradParameters()
        self.criterion:backward(self.model.output, self.target)
        self.model:backward(inputTuple, self.criterion.gradInput)
        --require('fb.debugger'):enter()

        if self.opt.optimizer == 'sgd' then
            optim.sgd(feval, self.params, self.optimState)
        elseif self.opt.optimizer == 'adam' then
            optim.adam(feval, self.params, self.optimState)
        elseif self.opt.optimizer == 'rmsprop' then
            optim.rmsprop(feval, self.params, self.optimState)
        end

        local top1 = self:computeScore(output, self.target, 1)
        top1Sum = top1Sum + top1 * batchSize
        lossSum = lossSum + loss * batchSize
        N = N + batchSize

        print(('%s | Epoch: [%d][%d/%d],   Time %.3f,   DataTime %.3f,   Loss %1.4f,   Top1 error %7.3f'):format(
            opt.name, epoch, n, trainSize, timer:time().real, dataTime, loss, top1))

        -- check that the storage didn't get changed due to an unfortunate getParameters call
         assert(self.params:storage() == self.model:parameters()[1]:storage())

        timer:reset()
        dataTimer:reset()
    end

   return top1Sum/N, lossSum/N
end

function Trainer:test_top1(opt, epoch, dataloader)
    -- Computes the top-1 error on the validation set

    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local size = dataloader:size()

    local nCrops = self.opt.tenCrop and 10 or 1
    local top1Sum = 0.0
    local lossSum = 0.0
    local N = 0

    self.model:evaluate()
    for n, sample in dataloader:run(false) do
        local dataTime = dataTimer:time().real
        -- Copy input and target to the GPU
        self:copyInputs(sample)

        local batchSize = self.input:size(1)
        local timesteps = self.input:size(2)
        self.input = self.input:view(batchSize * timesteps, -1)
        self.target = self.target:view(batchSize * timesteps, -1)

        local Q_o = torch.rand(batchSize * timesteps, opt.nObjects):cuda()
        local Q_v = torch.rand(batchSize * timesteps, opt.nVerbs):cuda()
        local inputTuple = {self.input, Q_o, Q_v}
        local output = self.model:forward(inputTuple):float()
        local loss = self.criterion:forward(self.model.output, self.target)

        local top1 = self:computeScore(output, self.target, nCrops)
        top1Sum = top1Sum + top1 * batchSize
        lossSum = lossSum + loss * batchSize
        N = N + batchSize

        print(('%s | Test: [%d][%d/%d],   Time %.3f,   DataTime %.3f,   top1 %7.3f (%7.3f)'):format(
            opt.name, epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N))

        timer:reset()
        dataTimer:reset()
    end
    self.model:training()

    print((' * Finished epoch # %d   top1: %7.3f\n'):format(epoch, top1Sum / N))

    return top1Sum / N, lossSum / N
end

function Trainer:test_mAP(opt, epoch, dataloader)
   -- Computes the mAP over the whole video sequences

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = 1
   local N = 0
   local outputs = torch.Tensor(2000,157) --allocate memory
   local gt = torch.Tensor(2000,157) --allocate memory
   local names = {}

   local frameoutputs, framenr, framenames, nframe
   if opt.dumpLocalize then
       frameoutputs = torch.Tensor(25*2000,157)
       framenames = {}
       framenr = {}
       nframe = 0
   end

   self.model:evaluate()
   n2 = 0
   for n, sample in dataloader:run() do
      n2 = n2 + 1
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = 25

      for i=1,25-1 do -- make sure there is no error in the loader, this should be one video
          assert(torch.all(torch.eq(
              sample.target[{{i},{}}],
              sample.target[{{i+1},{}}]
          )))
      end

      local tmp = output:exp()
      tmp = tmp:cdiv(tmp:sum(2):expandAs(output))
      outputs[{{n2},{}}] = tmp:mean(1)
      gt[{{n2},{}}] = sample.target[{{1},{}}]
      table.insert(names,sample.ids[1])

      if opt.dumpLocalize then
          frameoutputs[{{nframe+1,nframe+25},{}}] = tmp
          for b=1,25 do
              framenames[nframe+b] = sample.ids[1]
              framenr[nframe+b] = b
          end
          nframe = nframe+25
      end

      print(('%s | Test2: [%d][%d/%d]    Time %.3f  Data %.3f'):format(
         opt.name, epoch, n, size, timer:time().real, dataTime))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()
   outputs = outputs[{{1,n2},{}}] 
   gt = gt[{{1,n2},{}}] 
   ap = charades_ap(outputs, gt)

   print((' * Finished epoch # %d     mAP: %7.3f\n'):format(
      epoch, torch.mean(ap)))

   print('dumping output to file')
   local out = assert(io.open(self.opt.save .. "/epoch" .. epoch .. ".txt", "w"))
   for i=1,outputs:size(1) do
      out:write(names[i] .. " " .. tensor2str(outputs[{{i},{}}]:view(-1)) .. "\n")  
   end
   out:close()

   if opt.dumpLocalize then
       print('dumping localization output to file')
       frameoutputs = frameoutputs[{{1,nframe},{}}] 
       local out = assert(io.open(self.opt.save .. "/localize" .. epoch .. ".txt", "w"))
       for i=1,frameoutputs:size(1) do
          f = framenr[i]
          vidid = framenames[i]
          out:write(vidid .. " " .. f .. " " .. tensor2str(frameoutputs[{{i},{}}]:view(-1)) .. "\n")  
       end
       out:close()
   end

   return ap
end

function Trainer:computeScore(output, target, nCrops)
    if nCrops > 1 then
        -- Sum over crops
        output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
            --:exp()
            :sum(2):squeeze(2)
    end

    -- Coputes the top1 error rate
    -- local output = output:float()
    local _, pred = torch.topk(output, 1, 2, true) -- descending
    pred = pred:view(pred:nElement())
    local correct = target:index(2, pred):diag()

    -- Top-1 score
    local top1 = 1.0 - torch.mean(correct)

    return top1 * 100
end

function Trainer:copyInputs(sample)
    -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
    -- if using DataParallelTable. The target is always copied to a CUDA tensor
    self.input = self.input or (self.opt.nGPU == 1
        and torch.CudaTensor()
        or cutorch.createCudaHostTensor())
    self.target = self.target or torch.CudaTensor()

    self.input:resize(sample.input:size()):copy(sample.input)
    self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRateModifier(epoch)
    -- Training schedule
    local decay = 0
    if self.opt.dataset == 'charades' then
        decay = math.floor((epoch - 1) / self.LR_decay_freq)
    elseif self.opt.dataset == 'charadesFeatures' then
        decay = math.floor((epoch - 1) / self.LR_decay_freq)
    elseif self.opt.dataset == 'imagenet' then
        decay = math.floor((epoch - 1) / 30)
    elseif self.opt.dataset == 'cifar10' then
        decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
    else
        decay = math.floor((epoch - 1) / self.LR_decay_freq)
    end
    return math.pow(0.1, decay)
end



return M.Trainer
