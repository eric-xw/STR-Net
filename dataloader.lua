--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--  
--  Contributor: Xin Wang

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
    -- The train and val loader
    local loaders = {}

    for i, split in ipairs{'train', 'val'} do
        local dataset = datasets.create(opt, split)
        loaders[i] = M.DataLoader(dataset, opt, split)
    end

    return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
    local manualSeed = opt.manualSeed
    local function init()
        require('datasets/' .. opt.dataset)
    end
    local function main(idx)
        if manualSeed ~= 0 then
            torch.manualSeed(manualSeed + idx)
        end
        torch.setnumthreads(1)
        _G.dataset = dataset
        return dataset:size()
    end

    local threads, sizes = Threads(opt.nThreads, init, main)
    self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
    self.threads = threads
    -- self.nInputDim = opt.nFeatures
    -- self.nClasses = opt.nClasses
    -- self.nVerbs = opt.nVerbs
    -- self.nObjects = opt.nObjects
    self.__size = sizes[1][1]
    self.split = split
    self.epochSize = tonumber(opt.epochSize)
    if self.epochSize and (self.epochSize < 1) then
        self.epochSize = torch.floor(self.epochSize * self.__size / opt.batchSize) * opt.batchSize
    end
    self.testSize = tonumber(opt.testSize)
    if self.testSize and (self.testSize < 1) then
        self.testSize = torch.floor(self.testSize * self.__size / opt.batchSize) * opt.batchSize
    end
    self.batchSize = math.floor(opt.batchSize / self.nCrops)
end

function DataLoader:size()
    if  self.split=='train' and self.epochSize and not (self.epochSize==1) then
        return math.ceil(self.epochSize / self.batchSize)
    elseif  self.split=='val' and self.testSize and not (self.testSize==1) then
        return math.ceil(self.testSize / self.batchSize)
    else
        return math.ceil(self.__size / self.batchSize)
    end
end

function DataLoader:run()
    print('DataLoader:run')
    local threads = self.threads
    local split = self.split
    local size, batchSize = self.__size, self.batchSize
    -- local nClasses = self.nClasses
    -- local nInputDim = self.nInputDim
    -- local nVerbs = self.nVerbs
    -- local nObjects = self.nObjects
    local perm = torch.randperm(size)

    if self.split=='train' then
        if self.epochSize and not (self.epochSize==1) then
            -- Ensure each sample is seen equally often
            -- but reduce the epochSize
            if not self.perm then 
                self.perm = torch.randperm(size) 
            end
            if self.perm:size(1) <= self.epochSize then
                self.perm = self.perm:cat(torch.randperm(size),1)
            end
            perm = self.perm[{{1,self.epochSize}}]
            self.perm = self.perm[{{self.epochSize+1,-1}}]
            size = self.epochSize
       else
            perm = torch.randperm(size)
       end
    elseif self.split=='val' then
        perm = torch.range(1,size)
        if self.testSize and not (self.testSize==1) then
            perm = perm[{{1,self.testSize}}]
            size = self.testSize
        end
    else
        assert(false,'split undefined')
    end

    local idx, sample = 1, nil
    local function enqueue()
        while idx <= size and threads:acceptsjob() do
            local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
            threads:addjob(
                function(indices, nCrops)
                    local sz = indices:size(1)
                    local batch, featureSize
                    local target, obj, verb
                    local ids = {}
                    local scene = torch.IntTensor(sz)
                    for i, idx in ipairs(indices:totable()) do
                        local sample = _G.dataset:get(idx)
                        local input = sample.input
                        if not batch then
                            featureSize = input:size():totable()
                            if nCrops > 1 then table.remove(featureSize, 1) end
                            batch = torch.FloatTensor(sz, nCrops, table.unpack(featureSize))
                        end
                        if not target then
                            local targetSize = sample.target:size():totable()
                            local verbSize = sample.verbLabel:size():totable()
                            local objSize = sample.objLabel:size():totable()
                            target = torch.IntTensor(sz, table.unpack(targetSize))
                            verb = torch.IntTensor(sz, table.unpack(verbSize))
                            obj = torch.IntTensor(sz, table.unpack(objSize))
                        end
                        batch[i]:copy(input)
                        target[i]:copy(sample.target)
                        obj[i]:copy(sample.objLabel)
                        verb[i]:copy(sample.verbLabel)
                        ids[i] = sample.id
                        scene[i] = sample.scene and sample.scene or 0
                    end
                    collectgarbage()
                    return {
                        input = batch:view(sz * nCrops, table.unpack(featureSize)),
                        target = target,
                        obj = obj,
                        verb = verb,
                        scene = scene,
                        ids = ids,
                    }
                end,
                function(_sample_)
                    sample = _sample_
                end,
                indices,
                self.nCrops
            )
            idx = idx + batchSize
        end
    end

    local n = 0
    local function loop()
        enqueue()
        if not threads:hasjob() then
            return nil
        end
        threads:dojob()
        if threads:haserror() then
            threads:synchronize()
        end
        enqueue()
        n = n + 1
        return n, sample
    end

    print('DataLoader:run finished.')
    return loop
end

return M.DataLoader
