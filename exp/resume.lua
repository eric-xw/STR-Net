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
'-netType','STR_Net',
'-dataset','charadesFeatures',
'-LR_decay_freq','30',
'-LR','0.001',
'-epochSize','1',
'-testSize','1',
'-nEpochs','30',
'-batchSize','1',
'-nThreads','4',
'-accumGrad','4',
'-retrain','./cache/flownet/checkpoints/model_9.t7', -- path to the trained model to use
'-epochNumber','9', -- what epoch to resume from
'-optimState','./cache/flowrgbnet/checkpoints/optimstate_9.t7', -- path to the optimizer state
'-cacheDir','./cache/',
'-rgb_data', '/mnt/sshd/xwang/charades/Charades_v1_features_rgb/',
'-flow_data', '/mnt/sshd/xwang/charades/Charades_v1_features_flow/',
'-trainfile','/mnt/sshd/xwang/charades/vu17_charades/Charades_vu17_train.csv',
'-testfile','/mnt/sshd/xwang/charades/vu17_charades/Charades_vu17_validation.csv',
'-optnet','true',
}
for _,v in pairs(morearg) do
    table.insert(arg,v)
end
dofile 'main.lua'