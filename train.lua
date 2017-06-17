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
        originalLR = opt.LR,
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
    self.model:zeroGradParameters()
    for n, sample in dataloader:run() do -- true: shuffle the data
        local dataTime = dataTimer:time().real

        -- Copy input and target to the GPU
        self:copyInputs(sample)
        local batchSize = self.input:size(1)

        local Q_o = torch.rand(batchSize, opt.nObjects):cuda()
        local Q_v = torch.rand(batchSize, opt.nVerbs):cuda()
        local inputTuple = {self.input, Q_o, Q_v}
        
        local output = self.model:forward(inputTuple)
        
        if opt.netType == 'STR_Net' then
            output = output:float()
            self.targetTuple = self.target
        elseif opt.netType == 'STR_parallel' then
            output = output[3]:float()
            self.targetTuple = {self.obj, self.verb, self.target}
        else
            assert(false, 'There is no such net:' .. opt.netType)
        end

        local loss = self.criterion:forward(self.model.output, self.targetTuple)
        table.insert(opt.train_loss_history, loss)
        
        self.criterion:backward(self.model.output, self.targetTuple)
        self.model:backward(inputTuple, self.criterion.gradInput)
        --require('fb.debugger'):enter()

        if n % opt.accumGrad == 0 then -- accumulate batches
            if self.opt.optimizer == 'sgd' then
                optim.sgd(feval, self.params, self.optimState)
            elseif self.opt.optimizer == 'adam' then
                optim.adam(feval, self.params, self.optimState)
            elseif self.opt.optimizer == 'rmsprop' then
                optim.rmsprop(feval, self.params, self.optimState)
            end
            self.model:zeroGradParameters()
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
    for n, sample in dataloader:run() do
        local dataTime = dataTimer:time().real
        -- Copy input and target to the GPU
        self:copyInputs(sample)
        local batchSize = self.input:size(1)

        local Q_o = torch.rand(batchSize, opt.nObjects):cuda()
        local Q_v = torch.rand(batchSize, opt.nVerbs):cuda()
        local inputTuple = {self.input, Q_o, Q_v}
        local output = self.model:forward(inputTuple)
        if opt.netType == 'STR_Net' then
            output = output:float()
        elseif opt.netType == 'STR_parallel' then
            output = output[3]:float()
        else
            assert(false, 'There is no such net:' .. opt.netType)
        end
        local loss = self.criterion:forward(self.model.output, self.target)

        local top1 = self:computeScore(output, self.target, nCrops)
        top1Sum = top1Sum + top1 * batchSize
        lossSum = lossSum + loss * batchSize
        N = N + batchSize

        print(('%s | Test(top1): [%d][%d/%d],   Time %.3f,   DataTime %.3f,   top1 %7.3f (%7.3f)'):format(
            opt.name, epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N))

        timer:reset()
        dataTimer:reset()
    end
    self.model:training()

    print((' * Finished epoch # %d   top1: %7.3f\n'):format(epoch, top1Sum / N))

    return top1Sum / N, lossSum / N
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

function Trainer:test_mAP(opt, epoch, dataloader)
    -- Computes the mAP over the whole video sequences

    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local size = dataloader:size()
    local nSegments = 25

    local video_output = torch.Tensor(size, opt.nClasses)
    local video_target = torch.Tensor(size, opt.nClasses)
    local frame_output = torch.Tensor(size * nSegments, opt.nClasses)
    local frame_target = torch.Tensor(size * nSegments, opt.nClasses)
    
    n2 = 0
    self.model:evaluate()
    for n, sample in dataloader:run() do
        n2 = n2 + 1
        local dataTime = dataTimer:time().real
        
        -- Copy input and target to the GPU
        self:copyInputs(sample)
        local batchSize = self.input:size(1) -- which is also the timesteps

        local Q_o = torch.rand(batchSize, opt.nObjects):cuda()
        local Q_v = torch.rand(batchSize, opt.nVerbs):cuda()
        local inputTuple = {self.input, Q_o, Q_v}
        local output = self.model:forward(inputTuple)
        
        if opt.netType == 'STR_Net' then
            output = output:float()
        elseif opt.netType == 'STR_parallel' then
            output = output[3]:float()
        else
            assert(false, 'There is no such net:' .. opt.netType)
        end

        local tmp = output:exp()
        tmp = tmp:cdiv(tmp:sum(2):expandAs(output))

        local keypoints = torch.linspace(1, batchSize, nSegments)
        local indices = torch.floor(keypoints) 
        local s = 1 + (n2 - 1) * nSegments
        local e = s + nSegments - 1
        frame_output[{{s, e},{}}] = tmp:index(1, indices:long())
        frame_target[{{s, e},{}}] = self.target:index(1, indices:long()):float()
        -- for i = 1, nSegments-1 do
        --     local index = torch.floor(keypoints[i])            
        --     local ii = i + (n2 - 1) * nSegments
        --     frame_output[{{ii},{}}] = tmp[index]
        --     frame_target[{{ii},{}}] = self.target[index]:float()
        -- end

        video_output[{{n2},{}}] = tmp:index(1, indices:long()):mean(1)
        video_target[{{n2},{}}] = self.target:float():sum(1):ne(0)

        print(('%s | Test(mAP): [%d][%d/%d]    Time %.3f  DataTime %.3f'):format(
            opt.name, epoch, n, size, timer:time().real, dataTime))

        timer:reset()
        dataTimer:reset()
    end
    self.model:training()

    local function get_mAP( ap )
        local nan_mask = ap:ne(ap)
        local notnan_mask = ap:eq(ap)
        ap[nan_mask] = 0
        local mAP = ap:sum() / notnan_mask:sum()
        return mAP
    end

    local class_ap = charades_ap(video_output, video_target)
    local class_mAP = get_mAP(class_ap)
    print((' * Finished epoch # %d     Classification mAP: %7.3f\n'):format(epoch, class_mAP))

    local localization_ap = charades_ap(frame_output, frame_target)
    local localization_mAP = get_mAP(localization_ap)
    print((' * Finished epoch # %d     localization mAP: %7.3f\n'):format(epoch, localization_mAP))

    return class_mAP, localization_mAP
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
    self.verb = self.verb or torch.CudaTensor()
    self.obj = self.obj or torch.CudaTensor()

    self.input:resize(sample.input:size()):copy(sample.input)
    self.target:resize(sample.target:size()):copy(sample.target)
    self.verb:resize(sample.verb:size()):copy(sample.verb)
    self.obj:resize(sample.obj:size()):copy(sample.obj)
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
