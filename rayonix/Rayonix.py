import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from .baseline_als import baseline_als


class Rayonix:
    def __init__(self):
        self.detd = 285
        self.tthGon = 36
        self.param = [-1.767, 5.036, 0.025, 1.701, -0.243, 0.205, 0.225, 0.851]
        #self.param = [0,0,0,0,0,0,0,0]

        self.file = ""

        self.SS = 165  # ????
        self.binning = 2  # detector's binning. 2x2 binning is set as default
        self.detSize = 4096//self.binning  # detector's size in pixels

        # Default values
        self.lmb_default, self.p_default = 1e6, 1e-7
        self.etaCutoff_default, self.tthCutoff_default = [0., 1.], [0., 1.]
        self.step_default = .02

    def addZero(self, n):
        if (n < 10):
            s = "0%d" % n
        else:
            s = "%d" % n
        return s

    def tif2Data(self, filepath):
        data = plt.imread(filepath)
        return data

    def getDetPos(self):
        """Based on getDetPos.m implemented on Matlab by Daniel Candeloro Cunha @ LNNano"""

        Rx = self.param[0]
        Rr = self.param[1]
        Rt = self.param[2]
        Da = self.param[3]
        Db = self.param[4]
        Alp = self.param[5]
        Py = self.param[6]
        Pz = self.param[7]
        # degrees to radians
        Alp = Alp*np.pi/180
        tthRad = self.tthGon*np.pi/180

        ex = np.array([1, 0, 0])
        er = np.array([0, np.sin(tthRad), np.cos(tthRad)])
        et = np.array([0, np.cos(tthRad), -np.sin(tthRad)])

        p = self.detd*er
        c = p + Rr*er + Rt*et + Rx*ex
        ec = c/np.linalg.norm(c)
        a1 = ex - np.dot(ex, ec)*ec
        a1 = a1/np.linalg.norm(a1)
        b1 = np.cross(ec, a1)
        b = -np.sin(Alp)*a1 + np.cos(Alp)*b1
        a = np.cos(Alp)*a1 + np.sin(Alp)*b1
        d = c + Da*a + Db*b
        d = d + np.array([0, Py, Pz])

        return np.array([a, b, d])

    def getRectangular(self):
        """Get orthogonal Cartesian (x,y,z) coordinates of every pixel on Rayonix detector. Based on Rayonix.m implemented on Matlab by Daniel Candeloro Cunha @ LNNano"""

        ss = 1.*self.SS/self.detSize
        Ss = 1.*(self.SS - ss)/2

        detPos = self.getDetPos()
        aa = np.matrix(detPos[0, ])
        bb = np.matrix(detPos[1, ])
        dd = detPos[2, ]

        # self.detSize x 1 matrix
        i = np.matrix(range(self.detSize)).transpose()
        va = np.dot(Ss - i*ss, aa)
        vb = np.dot(Ss - i*ss, bb)

        retPos = np.zeros(shape=(self.detSize, self.detSize, 3))
        retPos[:, :, 0] = dd[0] + va[:, 0].transpose() + vb[:, 0]  # xp
        retPos[:, :, 1] = dd[1] + va[:, 1].transpose() + vb[:, 1]  # yp
        retPos[:, :, 2] = dd[2] + va[:, 1].transpose() + vb[:, 2]  # zp

        return retPos

    def getSpherical(self):
        """Get spherical coordinates (eta,2*theta,R) of every pixel on Rayonix detector. Based on Rayonix.m implemented on matlab by Daniel Candeloro Cunha @LNNano"""

        retPos = self.getRectangular()

        sphPos = np.zeros(shape=(self.detSize, self.detSize, 3))
        Rp = np.sqrt(retPos[:, :, 0]**2 + retPos[:, :, 1]
                     ** 2 + retPos[:, :, 2]**2)
        sphPos[:, :, 0] = Rp
        sphPos[:, :, 1] = np.arccos(
            retPos[:, :, 2]/Rp)*180/np.pi  # theta in degrees
        # eta in degrees
        sphPos[:, :, 2] = np.arctan2(-retPos[:, :, 0],
                                     retPos[:, :, 1])*180/np.pi

        return sphPos

    def dataTrial(self, **kwargs):
        """Remove zeros and trials the data according to eta and 2*theta limits (etaCutoff and tthCuttoff)"""

        etaCutoff = kwargs.pop("etaCutoff", self.etaCutoff_default)
        tthCutoff = kwargs.pop("tthCutoff", self.tthCutoff_default)

        sphPos = self.getSpherical()
        tth = sphPos[:, :, 1]
        eta = sphPos[:, :, 2]
        I = self.tif2Data(self.file)

        # transform 2D matrixes eta, tth and I in 1D arrays
        eta = eta.reshape(-1)
        tth = tth.reshape(-1)
        I = I.reshape(-1)

        # eliminate zeros (points outside the circle corresponding to the detector)
        jNonzero = np.nonzero(I)
        tth = tth[jNonzero]
        eta = eta[jNonzero]
        I = I[jNonzero]

        # etaCutoff
        etaMin = np.min(eta)
        etaMax = np.max(eta)
        etaAmp = etaMax - etaMin

        etaMax = etaMin + etaAmp*max(etaCutoff)
        etaMin = etaMin + etaAmp*min(etaCutoff)
        criteria = (eta >= etaMin) & (eta <= etaMax)

        tth = tth[criteria]
        eta = eta[criteria]
        I = I[criteria]

        # tthCutoff
        tthMin = np.min(tth)
        tthMax = np.max(tth)
        tthAmp = tthMax - tthMin

        tthMax = tthMin + tthAmp*max(tthCutoff)
        tthMin = tthMin + tthAmp*min(tthCutoff)
        criteria = (tth >= tthMin) & (tth <= tthMax)

        tth = tth[criteria]
        eta = eta[criteria]
        I = I[criteria]

        print("\n'{:}': {:d} points read!".format(self.file, len(I)))

        return (tth, eta, I)

    def dataSort(self, x, *args):
        """Sort Numpy arrays x, y and args as ascending x"""

        jSorted = np.argsort(x)
        x = x[jSorted]
        output = (x,)
        for arg in args:
            try:
                arg = arg[jSorted]
                output += (arg,)
            except:
                pass

        return output

    def integrate(self, x, y):
        """Numerical integration y.dx using the trapezium rule"""

        x, y = self.dataSort(x, y)
        area = .5*np.sum((y[1:]+y[:-1])*np.diff(x))
        return(area)

    def getDiffractogram(self, **kwargs):
        """Generates a difratogram (I vs tth) by integrating intensities along eta"""

        step = kwargs.pop("step", self.step_default)
        etaCutoff = kwargs.pop("etaCutoff", self.etaCutoff_default)
        tthCutoff = kwargs.pop("tthCutoff", self.tthCutoff_default)
        fileOutput = kwargs.pop("fileOutput", None)

        tth, eta, I = self.dataTrial(etaCutoff=etaCutoff, tthCutoff=tthCutoff)
        tth, eta, I = self.dataSort(tth, eta, I)

        tthMean = []
        IInt = []

        t0 = time.time()
        js = 0
        sys.stdout.write("\rGetting diffractogram...\n")
        for j in range(len(I)):
            if (tth[j]-tth[js]) > step:
                jf = j
                seg = range(js, jf)
                js = j

                tthMean = np.append(tthMean, np.mean(tth[seg]))
                # normalize IInt by number of pixels computed
                IInt = np.append(IInt, np.sum(I[seg])/len(seg))

                sys.stdout.write("\rTime elapsed: %.5f seconds" %
                                 (time.time()-t0))
                sys.stdout.flush()

        sys.stdout.write("\n")

        if fileOutput != None:
            np.savetxt(fileOutput,
                       list(zip(tthMean, IInt)),
                       fmt="%.6e", header="tth\tI")

        return (tthMean, IInt)

    def getCorrectedDiffractogram(self, **kwargs):
        """Remove diffractogram baseline (background). Also return bck, useful for data normalization"""

        step = kwargs.pop("step", self.step_default)
        etaCutoff = kwargs.pop("etaCutoff", self.etaCutoff_default)
        tthCutoff = kwargs.pop("tthCutoff", self.tthCutoff_default)
        lmb = kwargs.pop("lmb", self.lmb_default)
        p = kwargs.pop("p", self.p_default)
        fileOutput = kwargs.pop("fileOutput", None)

        tth, I = self.getDiffractogram(
            step=step, etaCutoff=etaCutoff, tthCutoff=tthCutoff)
        bck = baseline_als(I, lmb=lmb, p=p)

        IBase = I - bck

        if fileOutput != None:
            np.savetxt(fileOutput,
                       list(zip(tth, I, IBase, bck)),
                       fmt="%.6e", header="tth\tI\tIBase\tbck")
            print("\n'{}' successfully created!".format(fileOutput))

        return (tth, I, IBase, bck)

    def normalizeSetDiffractograms(self, rootFilepath, n, **kwargs):
        """Normalize a set o diffractrogram (same rootFilepath) using the integrated area of the background.
        Instead of using the raw .tif files as input, this routine uses the output files generated after running getCorrectedDiffractogram.

        'rootFilepath' is the filepath to the files genenerated by getCorrectedDiffractogram and 'n' is the number of files per sample."""

        save = kwargs.pop("save", False)

        tth, I, IBase, bck, hst = [], [], [], [], []

        for i in range(1, n+1):
            filepath = "%s_%s.txt" % (rootFilepath, self.addZero(i))
            data = np.loadtxt(filepath)

            tth = np.append(tth, data[:, 0])
            I = np.append(I, data[:, 1])
            IBase = np.append(IBase, data[:, 2])
            bck = np.append(bck, data[:, 3])
            hst = np.append(hst, len(data[:, 0])*[i])

        bckArea = self.integrate(tth, bck)
        I /= bckArea
        IBase /= bckArea

        if save:
            # every histogram in one file
            fileOutput = "normalized/%s.txt" % rootFilepath.split("/")[-1]
            np.savetxt(fileOutput,
                       list(zip(tth, I, IBase, hst)),
                       fmt=["%.6e", "%.6e", "%.6e", "%d"],
                       header="tth\tI\tIBase\thst")
            print("\n'{}' successfully created!".format(fileOutput))

            # each histogram in separate files
            for i in range(1, n+1):
                fileOutput = "normalized/%s_%s.txt" % (
                    rootFilepath.split("/")[-1], self.addZero(i))
                criteria = (hst == i)
                np.savetxt(fileOutput,
                           list(zip(tth[criteria], I[criteria],
                                    IBase[criteria])),
                           fmt="%.6e", header="tth\tI\tIBase")
                print("\n'{}' successfully created!".format(fileOutput))

        return (tth, I, IBase, hst)

    def plotDetector(self, ax=None, fileOutput=None, f=lambda x: x, **kwargs):
        """Plot the detector image as a colormap. Options include plotting the square root and logarithm of I (sqrt=True and log=True)"""

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        I = self.tif2Data(self.file)

        colorbar = kwargs.pop("colorbar", True)

        img = ax.imshow(f(I), **kwargs)
        img.set_rasterized(True)

        if colorbar:
            plt.colorbar(img, ax=ax)

        if fileOutput:
            fig.savefig(fileOutput, dpi=300)

    def plotSpherical(self, ax=None, fileOutput=None, f=lambda x: x, **kwargs):
        """Plot the detector image as colormap using the spherical coordinates eta and 2*theta"""

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        sphPos = self.getSpherical()
        tth = sphPos[:, :, 1]
        eta = sphPos[:, :, 2]
        I = self.tif2Data(self.file)

        colorbar = kwargs.pop("colorbar", True)

        img = ax.pcolormesh(eta, tth, f(I), **kwargs)
        img.set_rasterized(True)

        ax.set_xlabel("eta")
        ax.set_ylabel("2*theta")

        if colorbar:
            plt.colorbar(img, ax=ax)

        if fileOutput:
            fig.savefig(fileOutput, dpi=300)
