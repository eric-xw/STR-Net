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
    self.__size = sizes[1][1]
    self.split = split
    self.batchSize = math.floor(opt.batchSize / self.nCrops)
end

function DataLoader:size()
    return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
    local threads = self.threads
    local split = self.split
    local size, batchSize = self.__size, self.batchSize
    local perm
    if opt.shuffle then
        perm = torch.randperm(size)
    else
        print('No suffle for testing ..')
        perm = torch.linspace(1, size, size) -- no random for testing
    end

    local idx, sample = 1, nil
    local function enqueue()
        while idx <= size and threads:acceptsjob() do
            local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
            threads:addjob(
                function(indices, nCrops)
                    local sz = indices:size(1)
                    local batch, featureSize
                    local targetSize = sample.target:size():totable()
                    local target = torch.IntTensor(sz, table.unpack(targetSize))
                    local ids = {}
                    local obj = torch.IntTensor(sz)
                    local verb = torch.IntTensor(sz)
                    local scene = torch.IntTensor(sz)
                    for i, idx in ipairs(indices:totable()) do
                        local sample = _G.dataset:get(idx)
                        local input = sample.input
                        if not batch then
                            featureSize = input:size():totable()
                            if nCrops > 1 then table.remove(featureSize, 1) end
                            batch = torch.FloatTensor(sz, nCrops, table.unpack(featureSize))
                        end
                        batch[i]:copy(input)
                        target[i]:copy(sample.target)
                        ids[i] = sample.id
                        obj[i] = sample.obj and sample.obj or 0
                        verb[i] = sample.verb and sample.verb or 0
                        scene[i] = sample.scene and sample.scene or 0
                    end
                    collectgarbage()
                    return {
                        input = batch:view(sz * nCrops, table.unpack(featureSize)),
                        target = target,
                        ids = ids,
                        obj = obj,
                        verb = verb,
                        scene = scene,
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

    return loop
end

return M.DataLoader
