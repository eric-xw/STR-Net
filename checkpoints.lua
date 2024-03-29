--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local checkpoint = {}

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))

   return latest, optimState
end

local function modelscore(name, score)
   print('dumping score to file')
   local out = assert(io.open(name, "w"))
   out:write("train top1: " .. score[1] .. "\n")
   --out:write("train top5: " .. score[2] .. "\n")
   out:write("test top1: " .. score[2] .. "\n")
   --out:write("test top5: " .. score[4] .. "\n")
   out:write("classification mAP: " .. score[3] .. "\n")
   out:write("localization mAP: " .. score[4] .. "\n")
   out:close()
end

function checkpoint.save(epoch, model, optimState, isBestModel, opt, score)
  -- don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- create a clean copy on the CPU without modifying the original network
   model = deepCopy(model):float():clearState()

   local modelFile = 'model_' .. epoch .. '.t7'
   local optimFile = 'optimState_' .. epoch .. '.t7'

   modelscore(paths.concat(opt.save, string.format("model_%03d.txt",epoch)), score)
   torch.save(paths.concat(opt.save, modelFile), model)
   torch.save(paths.concat(opt.save, optimFile), optimState)
   torch.save(paths.concat(opt.save, 'latest.t7'), {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
   })
   modelscore(paths.concat(opt.save, 'latest.txt'), score)

   if isBestModel then
      torch.save(paths.concat(opt.save, 'model_best.t7'), model)
      modelscore(paths.concat(opt.save, 'model_best.txt'), score)
   end
end

function checkpoint.saveLossHistory(opt, epoch)
   print('save loss history into file')
   filename = paths.concat(opt.save, string.format("loss_history_%03d.txt", epoch))
   local out = assert(io.open(filename, "w"))
   for i=1, #opt.train_loss_history do
      out:write(opt.train_loss_history[i] .. " ")
   end
   out:close()
end

return checkpoint
