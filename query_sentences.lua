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
model = torch.load('/scratch/mu388/model.net','ascii')

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    if vocab[line[i]] == nil then error({code="vocab", word = line[i]}) end
  end
  return line
end

function query_sentences()
	while true do
	  print("Query: len word1 word2 etc")
	  local ok, line = pcall(readline)
	  if not ok then
	    if line.code == "EOF" then
	      break -- end loop
	    elseif line.code == "vocab" then
	      print("Word not in vocabulary:", line.word)
	    elseif line.code == "init" then
	      print("Start with a number")
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
	  	for i=2, #line do
	  		x[1] = vocab[line[i]]
	  		err, next_s, pred = unpack(model:forward({x,x,s}))
	  		g_replace_table(s,next_s)
	  		io.write(line[i])
	  		io.write(' ')
	  	end
	  	previous_predicted = torch.multinomial(torch.exp(pred[1]:float()), 1)[1]
	  	io.write(ivocab[previous_predicted])
	  	io.write(' ')
	  	for i=1, tonumber(line[1])-1 do
	  		x[1] = previous_predicted
	  		err, next_s, pred = unpack(model:forward({x,x,s}))
	  		g_replace_table(s,next_s)
		  	previous_predicted = torch.multinomial(torch.exp(pred[1]:float()), 1)[1]
	  		io.write(ivocab[previous_predicted])
	  		io.write(' ')
	  	end
	  	io.write('\n')
	    -- print("Thanks, I will print foo " .. line[1] .. " more times")
	    -- for i = 1, line[1] do io.write('foo ') end
	    -- io.write('\n')
	  end
	end
end

query_sentences()
