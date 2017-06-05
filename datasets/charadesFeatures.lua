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

function CharadesDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = opt.data
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function CharadesDataset:get(i)
   local path = ffi.string(self.imageInfo.imagePath[i]:data())
   local image = self:_loadImage(paths.concat(self.dir, path))
   local class = self.imageInfo.imageClass[i]
   local id = ffi.string(self.imageInfo.ids[i]:data())

   return {
      input = image,
      target = class,
      id = id
   }
end

function CharadesDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function CharadesDataset:size()
   return self.imageInfo.imageClass:size(1)
end

