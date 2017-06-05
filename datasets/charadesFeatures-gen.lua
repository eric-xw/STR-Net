--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of Charades filenames and classes rgb images
--
--  This version is different from charades-gen.lua as this loads videos one by one
--  To train models that require sequential data, such as LSTM
--
--  This generates a file gen/charadesFeatures.t7 which contains the list of all
--  Charades training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--
--  Contributor: Xin Wang

local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function parseCSV(filename)
    require 'csvigo'
    print(('Loading csv: %s'):format(filename))
    local all = csvigo.load{path=filename, mode='tidy'}
    local ids = all['id']
    local objectss = all['objects']
    local actionss = all['actions']
    local N = #ids
    local objLabels, actionLabels = {}, {}
    for i = 1,#ids do
        local id = ids[i]
        -- read object labels
        local objects = objectss[i]
        local objLabel = {}
        for obj in string.gmatch(objects, '([^;]+)') do -- split on ';'
            table.insert(objLabel, obj)
        end
        objLabels[id] = objLabel
        -- read action labels
        local actions = actionss[i]
        local actionLabel = {}
        for a in string.gmatch(actions, '([^;]+)') do -- split on ';'
            local a = string.gmatch(a, '([^ ]+)') -- split on ' '
            table.insert(actionLabel,{c=a(), s=tonumber(a()), e=tonumber(a())})
        end
        actionLabels[id] = actionLabel
    end
    return objLabels, actionLabels
end

local function subrange(t, first, length)
    local sub = {}
    local last = first + length - 1
    for i = first, last do
        sub[#sub + 1] = t[i]
    end
    return sub
end


local function strings2tensor(strTable)
-- convert a table of paths to a tensor
    local num = #strTable
    local maxLength = -1
    for _, str in pairs(strTable) do
        maxLength = math.max(maxLength, #str + 1)
    end
    local res = torch.CharTensor(num, maxLength):zero()
    for i, str in ipairs(strTable) do
        ffi.copy(res[i]:data(), str)
    end
    return res
end

local function prepare(opt,labels,split)
    require 'sys'
    require 'string'
    -- local rgbPath = torch.CharTensor()
    -- local flowPath = torch.CharTensor()
    -- local imageClass = torch.ByteTensor()
    local rgbDir = opt.rgb_data
    local flowDir = opt.flow_data
    assert(paths.dirp(rgbDir), 'directory not found: ' .. rgbDir)
    assert(paths.dirp(flowDir), 'directory not found: ' .. flowDir)
    local rgbPathSegments, flowPathSegments, imageClassSegments, ids = {}, {}, {}, {}
    local FPS, testGAP = 24, 6
    local e = 0
    local count = 0
    
    for id,label in pairs(labels) do -- for each video, whose ID is
        e = e + 1
        if e % 100 == 1 then print('#videos: ' .. e) end
        iddir = rgbDir .. '/' .. id
        local f = io.popen(('find -L %s -iname "*.txt" '):format(iddir))
        if not f then 
            print('class not found: ' .. id)
            print(('find -L %s -iname "*.txt" '):format(iddir))
        else
            local lines = {}
            while true do -- read all frame files into a table 
                local line = f:read('*line')
                if not line then break end
                table.insert(lines,line)
            end
            local N = #lines

            if opt.setup == 'softmax' then
                local local_rgbPaths = {}
                local local_flowPaths = {}
                local local_imageClasses = torch.zeros(N, 157):byte()
                if #label>0 then 
                    -- To generate training data with softmax loss (only one label)
                    -- We create a sorted pool with all pairs of (frames,label) 
                    -- and then randomly select a subset of those according to our batch size
                    -- Someone should really figure out how to properly use sigmoid loss for this
                    for i = 1, N do 
                        for _,anno in pairs(label) do
                            if (anno.s<(i-1)/FPS) and ((i-1)/FPS<anno.e) then
                                local a = 1+tonumber(string.sub(anno.c,2,-1))
                                local_imageClasses[i][a] = 1
                            end
                        end
                        local rgbPath = rgbDir .. '/' .. id .. '/' .. '/' .. id .. '-' .. string.format('%06d',i) .. '.txt'
                        local flowPath = flowDir .. '/' .. id .. '/' .. '/' .. id .. '-' .. string.format('%06d',i) .. '.txt'
                        table.insert(local_rgbPaths,rgbPath)
                        table.insert(local_flowPaths,flowPath)
                    end
                end
                local frameNum = #local_rgbPaths
                if frameNum >= opt.timesteps then
                    segmentNum = frameNum / opt.timesteps
                    for i = 1, segmentNum do
                        count = count + 1
                        local index = 1 + (i - 1) * opt.timesteps
                        local rgb_segment = subrange(local_rgbPaths, index, opt.timesteps)
                        local flow_segment = subrange(local_flowPaths, index, opt.timesteps)
                        local label_segment = local_imageClasses:narrow(1, index, opt.timesteps)
                        
                        table.insert(rgbPathSegments, strings2tensor(rgb_segment))
                        table.insert(flowPathSegments, strings2tensor(flow_segment))
                        table.insert(imageClassSegments, label_segment)
                        table.insert(ids, id)
                    end
                    remain = frameNum % opt.timesteps
                    if remain > 0.33 * opt.timesteps then
                        -- TODO
                    end
                end
            elseif opt.setup == 'sigmoid' then
                -- TODO
                assert(false,'Invalid opt.setup')
            else
                assert(false,'Invalid opt.setup')
            end
            f:close()
        end
    end

    -- Convert the generated list to a tensor for faster loading
    local classTensor = torch.cat(imageClassSegments, 1):view(count, opt.timesteps, -1)
    local idsTensor = strings2tensor(ids)

    return rgbPathSegments, flowPathSegments, classTensor, idsTensor
end


local function findClasses(dir)
   return Nil, Nil
end


function M.exec(opt, cacheFile)
    
    local filename = opt.trainfile
    local filenametest = opt.testfile 
    local _, labels = parseCSV(filename)
    print('done parsing train csv')
    local _, labelstest = parseCSV(filenametest)
    print('done parsing test csv')

    print("=> Generating list of videos/frames")
    local classList, classToIdx = findClasses(trainDir)

    print(" | finding all validation videos")
    local val_rgbPath, val_flowPath, val_featureClass, val_ids = prepare(opt,labelstest,'val')

    print(" | finding all training videos")
    local train_rgbPath, train_flowPath, train_featureClass, train_ids = prepare(opt,labels,'train')

    local info = {
        rgbDir = opt.rgb_data,
        rgbDir = opt.flow_data,
        classList = classList,
        train = {
            rgbPath = train_rgbPath, -- a table of segments, each one is a 2D CharTensors (frame, path)
            flowPath = train_flowPath, 
            featureClass = train_featureClass, -- 3D ByteTensor (segment, frame, class), each frame has a ByteTensor of size 157
            ids = train_ids
        },
        val = {
            rgbPath = val_rgbPath, -- a table of segments, each one is a 2D CharTensors (frame, path)
            flowPath = val_flowPath, 
            featureClass = val_featureClass, -- 3D ByteTensor (segment, frame, class), each frame has a ByteTensor of size 157
            ids = val_ids
        },
   }

   print(" | saving list of videos/frames to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
