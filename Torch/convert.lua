require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'sys'
local DataLoader = require 'dataloader'
local models = require 'networks/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

io.write(" * End of Training Process... ")

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.best(opt)

print("Compressing "..sys.COLORS.cyan .. checkpoint.modelFile .. sys.COLORS.none .. '\n')

-- Create model
local model, criterion = models.setup(opt, checkpoint)

large_file = paths.concat(opt.save, checkpoint.modelFile)
small_file = paths.concat(opt.save, checkpoint.modelFile)

local function split_mem(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i+1
    end
    return t
end

print("\n===================== Memory status Report ======================")
io.write("  * The disk space for model before compression: ")
bef = sys.execute('du -sh ' .. large_file)
bef = split_mem(bef, '\t')
print('\t'..bef[1])

os.remove(large_file)
torch.save(small_file, model:clearState())

io.write("  * The disk space for model after compression:  ")
res = sys.execute('du -sh ' .. small_file)
res = split_mem(res, '\t')
print('\t'..res[1])
print("=================================================================")
