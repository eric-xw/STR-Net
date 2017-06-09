--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Charades dataset loader
--  Contributor: Xin Wang

local image = require 'image'
local paths = require 'paths'
local ffi = require 'ffi'

local M = {}
local CharadesDataset = torch.class('resnet.CharadesDataset', M)

function CharadesDataset:__init(info, opt, split)
   self.info = info[split]
   self.opt = opt
   self.split = split
   self.rgbDir = opt.rgb_data
   self.flowDir = opt.flow_data
   assert(paths.dirp(self.rgbDir), 'directory does not exist: ' .. self.rgbDir)
   assert(paths.dirp(self.flowDir), 'directory does not exist: ' .. self.flowDir)
end

function CharadesDataset:get(i)
   local feature = self:_loadFeature(self.info.rgbPath[i], self.info.flowPath[i])
   local class = self.info.featureClass[i]
   local id = ffi.string(self.info.ids[i]:data())

   return {
      input = feature,
      target = class,
      id = id
   }
end

function CharadesDataset:_loadFeature(rgbPaths, flowPaths)
   
   local function loadFile(path)
      --print(path)
      local file = io.open(path)
      local feature = torch.Tensor(file:lines()():split(' '))
      file:close()
      return feature
   end

   local rgbFeatures = {}
   local flowFeatures = {}

   local frameNum = rgbPaths:size(1)
   for i = 1, frameNum do
      local rgbPath = ffi.string(rgbPaths[i]:data())
      local flowPath = ffi.string(flowPaths[i]:data())
      table.insert(rgbFeatures, loadFile(rgbPath))
      table.insert(flowFeatures, loadFile(flowPath))
   end

   local rgb = torch.cat(rgbFeatures):view(frameNum, -1)
   local flow = torch.cat(flowFeatures):view(frameNum, -1)

   return torch.cat(rgb, flow, 2)
end

function CharadesDataset:size()
   return self.info.featureClass:size(1)
end

return M.CharadesDataset
