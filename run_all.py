from trainval_net import trainval_net
from test_net import test_net
from options import AllOptions

if __name__ == "__main__":
	opt = AllOptions().parse()
	#trainval_net(opt)
	test_net(opt)