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
'-netType','STR_parallel',
'-dataset','charadesFeatures',
'-LR_decay_freq','30',
'-LR','0.001',
'-epochSize','1',
'-testSize','1',
'-nEpochs','30',
'-batchSize','1',
'-nThreads','4',
'-accumGrad','4',
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
