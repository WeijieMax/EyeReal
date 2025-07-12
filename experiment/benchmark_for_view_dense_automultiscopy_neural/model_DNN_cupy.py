from chainer import Chain
import chainer.functions as F
import chainer.links as L


    

class CalcLayer(Chain):
    def __init__(self):
        super(CalcLayer, self).__init__(
            cn01 = L.Convolution2D(25,64,3,pad=1),
            cn02 = L.Convolution2D(64,64,3,pad=1),
            cn03 = L.Convolution2D(64,64,3,pad=1),
            cn04 = L.Convolution2D(64,64,3,pad=1),
            cn05 = L.Convolution2D(64,64,3,pad=1),
            cn06 = L.Convolution2D(64,64,3,pad=1),
            cn07 = L.Convolution2D(64,64,3,pad=1),
            cn08 = L.Convolution2D(64,64,3,pad=1),
            cn09 = L.Convolution2D(64,64,3,pad=1),
            cn10 = L.Convolution2D(64,64,3,pad=1),
            cn11 = L.Convolution2D(64,64,3,pad=1),
            cn12 = L.Convolution2D(64,64,3,pad=1),
            cn13 = L.Convolution2D(64,64,3,pad=1),
            cn14 = L.Convolution2D(64,64,3,pad=1),
            cn15 = L.Convolution2D(64,64,3,pad=1),
            cn16 = L.Convolution2D(64,64,3,pad=1),
            cn17 = L.Convolution2D(64,64,3,pad=1),
            cn18 = L.Convolution2D(64,64,3,pad=1),
            cn19 = L.Convolution2D(64,64,3,pad=1),
            cnL = L.Convolution2D(64,3,3,pad=1),
        )

    def __call__(self, x):
        f = F.relu(self.cn01(x))
        f = F.relu(self.cn02(f))
        f = F.relu(self.cn03(f))
        f = F.relu(self.cn04(f))
        f = F.relu(self.cn05(f))
        f = F.relu(self.cn06(f))
        f = F.relu(self.cn07(f))
        f = F.relu(self.cn08(f))
        f = F.relu(self.cn09(f))
        f = F.relu(self.cn10(f))
        f = F.relu(self.cn11(f))
        f = F.relu(self.cn12(f))
        f = F.relu(self.cn13(f))
        f = F.relu(self.cn14(f))
        f = F.relu(self.cn15(f))
        f = F.relu(self.cn16(f))
        f = F.relu(self.cn17(f))
        f = F.relu(self.cn18(f))
        f = F.relu(self.cn19(f))
        f = self.cnL(f)
        return F.relu(f)
