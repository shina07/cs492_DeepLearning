import os, sys, time

hidden_units, batch_size, learning_rate, steps, dropout_rate = sys.argv[1:]
filename = 'result_%s_%s_%s_%s_%s.txt' % (hidden_units, batch_size, learning_rate, steps, dropout_rate)

with open(filename, 'w') as f:
	f.write("\n")
	f.write('-----Arguments---------------------\n')
	f.write('hidden layer units : %s\n' % hidden_units)
	f.write('batch size         : %s\n' % batch_size)
	f.write('learning rate      : %s\n' % learning_rate)
	f.write('steps              : %s\n' % steps)
	f.write('dropout rate       : %s\n' % dropout_rate)
	f.write('-----------------------------------\n\n')

	user_input = " ".join (sys.argv[1:])
	start_time = 0.0
	end_time = 0.0

	for i in (3, 5, 7):
		f.write('[depth %d]\n' % i)
		argv = str(i) + " " + user_input

		for j in range(1):
			print ("depth : %d, interation : %d\n" % (i, (j + 1)))
			f.write('    [iteration %d]\n' % (j + 1))
			start_time = time.time ()
			f.write(os.popen('python3 skeleton.py ' + argv + ' 1>log.txt 2>&1').read())
			end_time = time.time ()
			f.write(os.popen('grep -n log.txt -e "acc"').read())
			f.write ("execution time : %f sec\n" % (end_time - start_time))
			f.write(os.popen('rm -rf model/').read())

os.system('cat %s' % filename)

