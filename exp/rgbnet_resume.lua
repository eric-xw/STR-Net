--  Action recognition experiment using rgb
-- 
--  Purpose: ?
--  
--  start torch
--  Usage: dofile 'exp/rgbnet.lua'

local info = debug.getinfo(1,'S');
name = info.source
name = string.sub(name,1,#name-4) --remove ext
local name = name:match( "([^/]+)$" ) --remove folders
arg = arg or {}
morearg = {
'-name',name,
'-netType','vgg16',
'-dataset','charades',
'-LR_decay_freq','30',
'-LR','0.001',
'-epochSize','0.1',
'-testSize','0.1',
'-nEpochs','10',
'-conv1LR','1',
'-conv2LR','1',
'-conv3LR','1',
'-conv4LR','1',
'-conv5LR','1',
'-batchSize','64',
'-accumGrad','4',
'-retrain','./cache/flownet/checkpoints/model_9.t7', -- path to the trained model to use
'-epochNumber','9', -- what epoch to resume from
'-optimState','./cache/flowrgbnet/checkpoints/optimstate_9.t7', -- path to the optimizer state
'-cacheDir','./cache/',
'-data','/mnt/sshd/xwang/charades/Charades_v1_flow/',
'-trainfile','/mnt/sshd/xwang/charades/vu17_charades/Charades_vu17_train.csv',
'-testfile','/mnt/sshd/xwang/charades/vu17_charades/Charades_vu17_validation.csv',
'-optnet','true',
}
for _,v in pairs(morearg) do
    table.insert(arg,v)
end
dofile 'main.lua'
