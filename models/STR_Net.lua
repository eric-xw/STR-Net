
require 'torch'
require 'nn'

function inference_block()
-- X, Q_v', Q_o' = unpack(net:forward(X, Q_o, Q_v))

    local nInputX = 4096
    local nInputQ_o = 200
    local nInputQ_v = 150
    
    local function basic(nInput1, nInput2, nOutput)
        local net = nn.Sequential()
            :add(nn.ParallelTable()
                :add(nn.Linear(nInput1, nOutput))
                :add(nn.Linear(nInput2, nOutput)))
            :add(nn.CAddTable(true))
            :add(nn.Sigmoid())
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
    nUnits = opt.unitNum or 8
    
    local net = nn.Sequential()
    
    local unit = inference_block()
    for i=1, nBlocks do
        if opt.share then
            net:add(unit)
        else
            net:add(inference_block())
        end
    end
    
    net:add(nn.NarrowTable(2,2))
    net:add(nn.JoinTable(1))
    
    return net
end

return createModel


