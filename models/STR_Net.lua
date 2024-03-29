require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'

local activation = nn.ReLU

local function inference_block(opt)
-- X, Q_o', Q_v' = unpack(net:forward(X, Q_o, Q_v))

    local nInputX = opt.nFeatures or 4096
    local nInputQ_o = opt.nObjects or 38
    local nInputQ_v = opt.nVerbs or 33
    
    local function basic(nInput1, nInput2, nOutput)
        local net = nn.Sequential()
            :add(nn.ParallelTable()
                :add(nn.Linear(nInput1, nOutput))
                :add(nn.Linear(nInput2, nOutput)))
            :add(nn.CAddTable(true))
            -- :add(nn.BatchNormalization(nOutput))
            :add(activation())
        return net
    end
    
    local net = nn.Sequential()
        :add(nn.ConcatTable() 
            :add(nn.SelectTable(1))
            :add(nn.ConcatTable()
                :add(nn.Sequential()
                    :add(nn.ConcatTable()
                        :add(nn.SelectTable(1))
                        :add(nn.SelectTable(3)))
                    :add(basic(nInputX, nInputQ_v, nInputQ_o)))
                :add(nn.Sequential()
                    :add(nn.NarrowTable(1,2))
                    :add(basic(nInputX, nInputQ_o, nInputQ_v)))))
        :add(nn.FlattenTable())
    return net
end

function createModel(opt)
    local nUnits = opt.unitNum or 8
    local inputSize = opt.nObjects + opt.nVerbs
    local lstmOutputSize = 256

    local model = nn.Sequential()
    
    -- Inference units
    local unit = inference_block(opt)
    -- local softmaxTable = nn.ParallelTable()
    --     :add(nn.Identity())
    --     :add(nn.SoftMax())
    --     :add(nn.SoftMax())
    for i=1, nUnits do
        if opt.share then
            model:add(unit)
        else
            model:add(inference_block(opt))
        end
        -- model:add(softmaxTable)
    end
    model:add(nn.NarrowTable(2,2))
    model:add(nn.JoinTable(2))

    -- LSTM layer
    local lstm = cudnn.LSTM(inputSize, lstmOutputSize, 1, true) 
    model:add(nn.View(1, -1, inputSize))
    model:add(nn.Copy(nil, nil, true))
    model:add(lstm)
    -- Dropout layer
    if opt.dropout > 0 then 
        model:add(nn.Dropout(opt.dropout))
    end
    -- Last FC layer
    model:add(nn.View(-1, lstmOutputSize))
    model:add(nn.Linear(lstmOutputSize, opt.nClasses))
    model:cuda()
    
    --print(tostring(model))

    return model
end

return createModel


