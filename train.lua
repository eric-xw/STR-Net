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
--  Contributor: Gunnar Atli Sigurdsson

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
    self.model = model
    self.criterion = criterion
    self.optimState = optimState or {
        originalLR = opt.LR,
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        nesterov = true,
        dampening = 0.0,
        weightDecay = opt.weightDecay,
    }
    self.opt = opt
    self.params, self.gradParams = model:parameters()
    self.LR_decay_freq = opt.LR_decay_freq
end

function Trainer:train(opt, epoch, dataloader)
   -- Trains the model for a single epoch

    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local LRM = self:learningRateModifier(epoch) 
    self.optimState.learningRate = self.optimState.learningRate * LRM

    local function feval(i)
        return self.criterion.output, self.gradParams
    end

    local trainSize = dataloader:size()
    local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
    local N = 0

    print('=> Training epoch # ' .. epoch)
    -- set the batch norm to training mode
    self.model:training()
    self.model:zeroGradParameters()
    for n, sample in dataloader:run() do
        local dataTime = dataTimer:time().real

        -- Copy input and target to the GPU
        self:copyInputs(sample)
        local batchSize = self.input:size(1)
        local timesteps = self.input:size(2)
        self.input = self.input:view(batchSize * timesteps, -1)
        self.target = self.target:view(batchSize * timesteps, -1)

        local output = self.model:forward(self.input):float()
        local loss = self.criterion:forward(self.model.output, self.target)

        self.criterion:backward(self.model.output, self.target)
        self.model:backward(self.input, self.criterion.gradInput)
        --require('fb.debugger'):enter()

        if self.opt.optimizer == 'sgd' then
            optim.sgd(feval, self.params, self.optimState)
        elseif self.opt.optimizer == 'adam' then
            optim.adam(feval, self.params, self.optimState)
        elseif self.opt.optimizer == 'adamax' then
            optim.adamax(feval, self.params, self.optimState)
        elseif self.opt.optimizer == 'rmsprop' then
            optim.rmsprop(feval, self.params, self.optimState)
        end 

        N = N + batchSize

        print(('%s | Epoch: [%d][%d/%d]    Time %.3f  DataTime %.3f  Loss %1.4f'):format(
            opt.name, epoch, n, trainSize, timer:time().real, dataTime, loss))

        -- check that the storage didn't get changed do to an unfortunate getParameters call
        assert(self.params[1]:storage() == self.model:parameters()[1]:storage()) -- TODO this ok?

        timer:reset()
        dataTimer:reset()
    end

   return lossSum / N
end

-- Torch port of THUMOSeventclspr in THUMOS'15
local function mAP(conf, gt)
    local so,sortind = torch.sort(conf, 1, true) --desc order
    local tp = gt:index(1,sortind:view(-1)):eq(1):int()
    local fp = gt:index(1,sortind:view(-1)):eq(0):int()
    local npos = torch.sum(tp)

    fp = torch.cumsum(fp)
    tp = torch.cumsum(tp)
    local rec = tp:float()/npos
    local prec = torch.cdiv(tp:float(),(fp+tp):float())
    
    local ap = 0
    local tmp = gt:index(1,sortind:view(-1)):eq(1):view(-1)
    for i=1,conf:size(1) do
        if tmp[i]==1 then
            ap = ap+prec[i]
        end
    end
    ap = ap/npos

    return rec,prec,ap
end

local function charades_ap(outputs, gt)
   -- approximate version of the charades evaluation function
   -- For precise numbers, use the submission file with the official matlab script
   conf = outputs:clone()
   conf[gt:sum(2):eq(0):expandAs(conf)] = -math.huge -- This is to match the official matlab evaluation code. This omits videos with no annotations 
   ap = torch.Tensor(157,1)
   for i=1,157 do
       _,_,ap[{{i},{}}] = mAP(conf[{{},{i}}],gt[{{},{i}}])
   end
   return ap
end

local function tensor2str(x)
    str = ""
    for i=1,x:size(1) do
        if i == x:size(1) then
            str = str .. x[i]
        else
            str = str .. x[i] .. " "
        end
    end
    return str
end

function Trainer:test2(opt, epoch, dataloader)
   -- Computes the mAP over the whole videos

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
