stringx = require('pl.stringx')
require 'nngraph'
require 'nn'
require 'torch'
require 'cunn'
require 'io'
require 'base.lua'
data = require('data.lua')
data.traindataset(1)
vocab = data.vocab_map
ivocab = data.inverse_vocab_map
model = torch.load('/scratch/mu388/model_cha.net','ascii')

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  for i = 1,#line do
    if vocab[line[i]] == nil then error({code="vocab", word = line[i]}) end
  end
  return line
end

function query_chas()
	io.write('OK GO\n')
	io.flush()
	while true do
	  local ok, line = pcall(readline)
	  if not ok then
	    if line.code == "EOF" then
	      break -- end loop
	    elseif line.code == "vocab" then
	      print("Word not in vocabulary:", line.word)
	    else
	      print(line)
	      print("Failed, try again")
	    end
	  else
	  	s = {}
	  	for i=1, 4 do
	  		s[i] = torch.zeros(20,200):cuda()
	  	end
	  	x = torch.zeros(20):cuda()
	  	pred = torch.zeros(20,50):cuda()
	  	for i=1, #line do
	  		x[1] = vocab[line[i]]
	  		err, next_s, pred = unpack(model:forward({x,x,s}))
	  		g_replace_table(s,next_s)
	  	end
	  	for i=1,50 do 
	  		io.write(pred[1][i])
	  		io.write(' ')
	  	end	
	  	io.flush()
	  	io.write('\n')
	  end
	end
end

query_chas()
