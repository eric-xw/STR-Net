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
    for i = first, first + length - 1 do
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
    local FPS, GAP, testGAP = 24, 1, 6
    local e,count = 1, 1
    
    for id,label in pairs(labels) do -- for each video, whose ID is
        if e % 100 == 1 then print('video number:' .. e) end
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
                local local_imageClasses = {}
                local local_ids = {}
                if #label>0 then 
                    -- To generate training data with softmax loss (only one label)
                    -- We create a sorted pool with all pairs of (frames,label) 
                    -- and then randomly select a subset of those according to our batch size
                    -- Someone should really figure out how to properly use sigmoid loss for this
                    for i = 1,N,GAP do -- for each frame at index 1 + GAP * iter, iter = 0, 1, 2, ...
                        local imageClass = torch.zeros(157):byte()
                        for _,anno in pairs(label) do
                            if (anno.s<(i-1)/FPS) and ((i-1)/FPS<anno.e) then
                                local a = 1+tonumber(string.sub(anno.c,2,-1))
                                imageClass[a] = 1
                            end
                        end
                        local rgbPath = rgbDir .. '/' .. id .. '/' .. '/' .. id .. '-' .. string.format('%06d',i) .. '.txt'
                        local flowPath = flowDir .. '/' .. id .. '/' .. '/' .. id .. '-' .. string.format('%06d',i) .. '.txt'
                        table.insert(local_rgbPaths,rgbPath)
                        table.insert(local_flowPaths,flowPath)
                        table.insert(local_imageClasses, a) -- 1-index
                        table.insert(local_ids,id)
                    end
                end
                local frameNum = #local_rgbPaths
                if frameNum >= opt.timesteps then
                    segmentNum = frameNum / opt.timesteps
                    for i = 1, segmentNum do
                        local index = 1 + (i - 1) * opt.timesteps
                        local rgb_segment = subrange(local_rgbPaths)
                        local flow_segment = subrange(local_flowPaths)
                        local label_segment = subrange(local_imageClasses)
                        
                        rgbPathSegments[count] = strings2tensor(rgb_segment)
                        flowPathSegments[count] = strings2tensor(flow_segment)
                        imageClassSegments[count] = torch.ByteTensor(label_segment)
                        ids[count] = id
                    end
                    remain = frameNum % opt.timesteps
                    if remain > 0.33 * opt.timesteps then
                        -- TODO
                    end
                end
                        -- table.insert(imagePaths,localimagePaths[a])
                        -- table.insert(imageClasses, localimageClasses[a]) -- 1-index
                        -- table.insert(ids,localids[a])

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
    ids_tensor = torch.CharTensor(ids)

    return rgbPathSegments, flowPathSegments, imageClassSegments, ids_tensor
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

    print("=> Generating list of images")
    local classList, classToIdx = findClasses(trainDir)

    print(" | finding all validation images")
    local val_rgbPath, val_flowPath, val_imageClass, val_ids = prepare(opt,labelstest,'val')

    print(" | finding all training videos")
    local train_rgbPath, train_flowPath, train_imageClass, train_ids = prepare(opt,labels,'train')

    local info = {
        basedir = opt.data,
        classList = classList,
        train = {
            rgbPath = train_rgbPath, -- a table of segments, each one is a 2D CharTensors (frame, path)
            flowPath = train_flowPath, 
            imageClass = train_imageClass, -- a table of segments, each one is 2D ByteTensor (frame, class), each frame has a ByteTensor of size 157
            ids = train_ids
        },
        val = {
            rgbPath = val_rgbPath, -- a table of segments, each one is a 2D CharTensors (frame, path)
            flowPath = val_flowPath, 
            imageClass = val_imageClass, -- a table of segments, each one is 2D ByteTensor (frame, class), each frame has a ByteTensor of size 157
            ids = val_ids
        },
   }

   print(" | saving list of videos/frames to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M