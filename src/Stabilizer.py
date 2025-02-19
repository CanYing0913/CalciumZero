import numpy as np
from math import sqrt
from tifffile import imread, imwrite
from tqdm import trange
from functools import lru_cache


# Note: getPixels (aka np.zeros_like) should be flatten.  -- 2023.07.27


class Stabilizer(object):
    __slots__ = [
        'transform',
        'pyramidLevel',
        'maxIter',
        'tol',
        'alpha',
        'img',
        'out',
        'ip',
        'ipRef'
    ]

    def __init__(self):
        self.transform = 'translation'
        self.pyramidLevel = 1
        self.maxIter = 200
        self.tol = 1E-7
        self.alpha = 0.9

        self.img = None
        self.out = None
        self.ip = None
        self.ipRef = None

    def run(self, path_input, path_output=None):
        self.img = imread(path_input)
        self.ipRef = np.asfortranarray(self.img[0])
        stackSize = len(self.ipRef)
        current = 0
        self.showProgress(0.0)
        self.process(current - 1, 1, -1, 1)
        self.process(current, stackSize, 1, current)
        if path_output is not None:
            imwrite(path_output, self.out)

    @staticmethod
    def showProgress(percent):
        print(f'{percent}')

    def showStatus(self, msg):
        print(f'{msg}')
        pass

    def process(self, firstslice: int,
                lastslice: int, interval: int, tick: int):
        stacksize, width, height = self.img.shape
        ipPyramid = [None, None, None, None, None]
        ipRefPyramid = [None, None, None, None, None]
        ipPyramid[0] = np.zeros((width, height), dtype=float)
        ipRefPyramid[0] = np.zeros((width, height), dtype=float)
        if self.pyramidLevel >= 1 and width >= 100 and height >= 100:
            width2 = width // 2
            height2 = height // 2
            ipPyramid[1] = np.zeros((width2, height2), dtype=float)
            ipRefPyramid[1] = np.zeros((width2, height2), dtype=float)
            if self.pyramidLevel >= 2 and width >= 200 and height >= 200:
                width4 = width // 4
                height4 = height // 4
                ipPyramid[2] = np.zeros((width4, height4), dtype=float)
                ipRefPyramid[2] = np.zeros((width4, height4), dtype=float)
                if self.pyramidLevel >= 3 and width >= 400 and height >= 400:
                    width8 = width // 8
                    height8 = height // 8
                    ipPyramid[3] = np.zeros((width8, height8), dtype=float)
                    ipRefPyramid[3] = np.zeros((width8, height8), dtype=float)
                    if self.pyramidLevel >= 4 and width >= 800 and height >= 800:
                        width16 = width // 16
                        height16 = height // 16
                        ipPyramid[4] = np.zeros((width16, height16), dtype=float)
                        ipRefPyramid[4] = np.zeros((width16, height16), dtype=float)

        for slice_idx in trange(firstslice, lastslice, interval):
            if slice_idx == firstslice and interval > 0:
                self.showStatus(f'skipping slice={slice_idx}')
                if self.out is None:
                    self.out = np.zeros_like(self.img)
                    self.out[0] = self.ipRef
                else:
                    raise NotImplementedError
                self.showProgress(tick / stacksize)
                tick += 1
            else:
                self.showStatus(f'stabilizing slice={slice_idx}')
                ip = self.img[slice_idx].copy()
                wp = [[]]
                if self.transform == 'translation':
                    wp = self.estimateTranslation1(
                        ip, ipPyramid, ipRefPyramid)
                else:
                    wp = self.estimateAffine1(
                        ip, ipPyramid, ipRefPyramid)
                ipOut = np.zeros((width, height), dtype=float)
                if self.transform == 'translation':
                    ipOut = self.warpTranslation(ipOut, ip, wp)
                else:
                    ipOut = self.warpAffine(ipOut, ip, wp)

                # Note: Skipped RGB image process
                # TODO line 299-318

                self.out[slice_idx] = ipOut

                self.ipRef = self.combine(self.ipRef, ipOut)
                self.showProgress(tick / stacksize)
                tick += 1

    def estimateAffine1(self, ip, ipPyramid, ipRefPyramid):
        ipRef = self.ipRef
        wp = np.array([[0.0, 0.0, 0.0] for _ in range(2)])
        ipPyramid[0] = self.gradient(ip)
        ipRefPyramid[0] = self.gradient(ipRef)
        if ipPyramid[4] is not None and ipRefPyramid[4] is not None:
            ipPyramid[4] = self.resize(ipPyramid[4], ipPyramid[0])
            ipRefPyramid[4] = self.resize(ipRefPyramid[4], ipRefPyramid[0])
            wp = self.estimateAffine2(wp, ipPyramid[4], ipRefPyramid[4])
            wp[0, 2] *= 16
            wp[1, 2] *= 16

        if ipPyramid[3] is not None and ipRefPyramid[3] is not None:
            ipPyramid[3] = self.resize(ipPyramid[3], ipPyramid[0])
            ipRefPyramid[3] = self.resize(ipRefPyramid[3], ipRefPyramid[0])
            wp = self.estimateAffine2(wp, ipPyramid[3], ipRefPyramid[3])
            wp[0, 2] *= 8
            wp[1, 2] *= 8

        if ipPyramid[2] is not None and ipRefPyramid[2] is not None:
            ipPyramid[2] = self.resize(ipPyramid[2], ipPyramid[0])
            ipRefPyramid[2] = self.resize(ipRefPyramid[2], ipRefPyramid[0])
            wp = self.estimateAffine2(wp, ipPyramid[2], ipRefPyramid[2])
            wp[0, 2] *= 4
            wp[1, 2] *= 4

        if ipPyramid[1] is not None and ipRefPyramid[1] is not None:
            ipPyramid[1] = self.resize(ipPyramid[1], ipPyramid[0])
            ipRefPyramid[1] = self.resize(ipRefPyramid[1], ipRefPyramid[0])
            wp = self.estimateAffine2(wp, ipPyramid[1], ipRefPyramid[1])
            wp[0, 2] *= 2
            wp[1, 2] *= 2

        wp = self.estimateAffine2(wp, ipPyramid[0], ipRefPyramid[0])

        return wp

    def estimateAffine2(self, wp, ip, ipRef):
        maxIter, tol = self.maxIter, self.tol
        width, height = ip.shape
        jx = np.zeros(width * height, dtype=float)
        jy = np.zeros(width * height, dtype=float)

        for y in range(height):
            for x in range(width):
                jx[y * width + x] = float(x)
                jy[y * width + x] = float(y)

        sd = np.array([[] for _ in range(6)])
        sd[4] = self.dx(ipRef)
        sd[5] = self.dy(ipRef)
        sd[0] = self.dot(sd[4], jx)
        sd[1] = self.dot(sd[5], jx)
        sd[2] = self.dot(sd[4], jy)
        sd[3] = self.dot(sd[5], jy)

        ipOut = ip.copy()

        dp = np.array([0.0 for _ in range(6)])

        bestWp = np.zeros((2, 3), dtype=float)
        bestWp[0, 0] = wp[0, 0]
        bestWp[0, 1] = wp[0, 1]
        bestWp[0, 2] = wp[0, 2]
        bestWp[1, 0] = wp[1, 0]
        bestWp[1, 1] = wp[1, 1]
        bestWp[1, 2] = wp[1, 2]

        d = np.identity(3, dtype=float)
        w = np.identity(3, dtype=float)
        h = np.zeros((6, 6), dtype=float)

        for y in range(6):
            for x in range(6):
                h[y, x] = self.dotSum(sd[x], sd[y])
        h = self.invert(h)
        oldRmse = float('inf')
        minRmse = float('inf')

        for iter_idx in range(maxIter):
            ipOut = self.warpAffine(ipOut, ip, wp)
            ipOut = self.subtract(ipOut, ipRef)
            rmse = self.rootMeanSquare(ipOut)
            if iter_idx > 0:
                if rmse < minRmse:
                    bestWp[0, 0] = wp[0, 0]
                    bestWp[0, 1] = wp[0, 1]
                    bestWp[0, 2] = wp[0, 2]
                    bestWp[1, 0] = wp[1, 0]
                    bestWp[1, 1] = wp[1, 1]
                    bestWp[1, 2] = wp[1, 2]
                    minRmse = rmse
                if abs((oldRmse - rmse) / (oldRmse + pow(2, -1074))) < tol:
                    break
            oldRmse = rmse

            error = ipOut

            dp[0] = self.dotSum(sd[0], error)
            dp[1] = self.dotSum(sd[1], error)
            dp[2] = self.dotSum(sd[2], error)
            dp[3] = self.dotSum(sd[3], error)
            dp[4] = self.dotSum(sd[4], error)
            dp[5] = self.dotSum(sd[5], error)

            dp = self.prod1(h, dp)

            d[0][0] = dp[0] + 1.0
            d[0][1] = dp[2]
            d[0][2] = dp[4]
            d[1][0] = dp[1]
            d[1][1] = dp[3] + 1.0
            d[1][2] = dp[5]
            d[2][0] = 0.0
            d[2][1] = 0.0
            d[2][2] = 1.0

            w[0][0] = wp[0][0] + 1.0
            w[0][1] = wp[0][1]
            w[0][2] = wp[0][2]
            w[1][0] = wp[1][0]
            w[1][1] = wp[1][1] + 1.0
            w[1][2] = wp[1][2]
            w[2][0] = 0.0
            w[2][1] = 0.0
            w[2][2] = 1.0

            w = self.prod2(w, self.invert(d))

            wp[0][0] = w[0][0] - 1.0
            wp[0][1] = w[0][1]
            wp[0][2] = w[0][2]
            wp[1][0] = w[1][0]
            wp[1][1] = w[1][1] - 1.0
            wp[1][2] = w[1][2]

        return bestWp

    def estimateTranslation1(self, ip, ipPyramid, ipRefPyramid):
        ipRef = self.ipRef
        wp = [[0.0] for _ in range(2)]
        ipPyramid[0] = self.gradient(ip)
        ipRefPyramid[0] = self.gradient(ipRef)

        if ipPyramid[4] is not None and ipRefPyramid[4] is not None:
            ipPyramid[4] = self.resize(ipPyramid[4], ipPyramid[0])
            ipRefPyramid[4] = self.resize(ipRefPyramid[4], ipRefPyramid[0])
            wp = self.estimateTranslation2(wp, ipPyramid[4], ipRefPyramid[4])
            wp[0][0] *= 16
            wp[1][0] *= 16

        if ipPyramid[3] is not None and ipRefPyramid[3] is not None:
            ipPyramid[3] = self.resize(ipPyramid[3], ipPyramid[0])
            ipRefPyramid[3] = self.resize(ipRefPyramid[3], ipRefPyramid[0])
            wp = self.estimateTranslation2(wp, ipPyramid[3], ipRefPyramid[3])
            wp[0][0] *= 8
            wp[1][0] *= 8

        if ipPyramid[2] is not None and ipRefPyramid[2] is not None:
            ipPyramid[2] = self.resize(ipPyramid[2], ipPyramid[0])
            ipRefPyramid[2] = self.resize(ipRefPyramid[2], ipRefPyramid[0])
            wp = self.estimateTranslation2(wp, ipPyramid[2], ipRefPyramid[2])
            wp[0][0] *= 4
            wp[1][0] *= 4

        if ipPyramid[1] is not None and ipRefPyramid[1] is not None:
            ipPyramid[1] = self.resize(ipPyramid[1], ipPyramid[0])
            ipRefPyramid[1] = self.resize(ipRefPyramid[1], ipRefPyramid[0])
            wp = self.estimateTranslation2(wp, ipPyramid[1], ipRefPyramid[1])
            wp[0][0] *= 2
            wp[1][0] *= 2

        wp = self.estimateTranslation2(wp, ipPyramid[0], ipRefPyramid[0])

        return wp

    def estimateTranslation2(self, wp, ip, ipRef):
        dxRef = self.dx(ipRef)
        dyRef = self.dy(ipRef)

        ipOut = ip.copy()

        dp = [0.0, 0.0]

        bestWp = np.zeros((2, 1), dtype=float)
        bestWp[0, 0] = wp[0, 0]
        bestWp[1, 0] = wp[1, 0]

        d = np.identity(3, dtype=float)
        w = np.identity(2, dtype=float)
        h = np.zeros((2, 2), dtype=float)

        h[0][0] = self.dotSum(dxRef, dxRef)
        h[1][0] = self.dotSum(dxRef, dyRef)
        h[0][1] = self.dotSum(dyRef, dxRef)
        h[1][1] = self.dotSum(dyRef, dyRef)
        h = self.invert(h)

        oldRmse = float('inf')
        minRmse = float('inf')
        dminv = pow(2, -1074)

        for iter in range(self.maxIter):
            ipOut = self.warpTranslation(ipOut, ip, wp)
            ipOut = self.subtract(ipOut, ipRef)
            rmse = self.rootMeanSquare(ipOut)
            if iter > 0:
                if rmse < minRmse:
                    bestWp[0, 0] = wp[0, 0]
                    bestWp[1, 0] = wp[1, 0]
                    minRmse = rmse
                if abs((oldRmse - rmse) / (oldRmse + dminv)) < self.tol:
                    break
            oldRmse = rmse

            error = ipOut

            dp[0] = self.dotSum(dxRef, error)
            dp[1] = self.dotSum(dyRef, error)

            dp = self.prod1(h, dp)

            d[0][0] = 1.0
            d[0][1] = 0.0
            d[0][2] = dp[0]
            d[1][0] = 0.0
            d[1][1] = 1.0
            d[1][2] = dp[1]
            d[2][0] = 0.0
            d[2][1] = 0.0
            d[2][2] = 1.0

            w[0][0] = 1.0
            w[0][1] = 0.0
            w[0][2] = wp[0][0]
            w[1][0] = 0.0
            w[1][1] = 1.0
            w[1][2] = wp[1][0]
            w[2][0] = 0.0
            w[2][1] = 0.0
            w[2][2] = 1.0

            w = self.prod2(w, self.invert(d))

            wp[0, 0] = w[0, 2]
            wp[1, 0] = w[1, 2]

        return bestWp

    def gradient(self, ip):
        width, height = ip.shape
        pixels = ip
        outPixels = np.zeros_like(ip, dtype=float)

        for y in range(1, height - 1):
            offset = 1 + y * width

            # nw---n---ne
            #  |   |   |
            #  w---o---e
            #  |   |   |
            # sw---s---se

            p2 = pixels[np.unravel_index(offset - width - 1, ip.shape)]  # nw
            p3 = pixels[np.unravel_index(offset - width, ip.shape)]  # n
            # p4 = 0.0  # ne
            p5 = pixels[np.unravel_index(offset - 1, ip.shape)]  # w
            p6 = pixels[np.unravel_index(offset, ip.shape)]  # o
            # p7 = 0.0  # e
            p8 = pixels[np.unravel_index(offset + width - 1, ip.shape)]  # sw
            p9 = pixels[np.unravel_index(offset + width, ip.shape)]  # s

            for x in range(1, width - 1):
                p1 = p2
                p2 = p3
                p3 = pixels[np.unravel_index(offset - width + 1, ip.shape)]
                p4 = p5
                p5 = p6
                p6 = pixels[np.unravel_index(offset + 1, ip.shape)]
                p7 = p8
                p8 = p9
                p9 = pixels[np.unravel_index(offset + width + 1, ip.shape)]
                a = p1 + 2 * p2 + p3 - p7 - 2 * p8 - p9
                b = p1 + 2 * p4 + p7 - p3 - 2 * p6 - p9
                outPixels[np.unravel_index(offset, ip.shape)] = sqrt(a * a + b * b)
                offset += 1

        return outPixels

    def resize(self, ipOut, ip):
        import threading
        width, height = ip.shape
        widthOut, heightOut = ipOut.shape
        xScale = width / widthOut
        yScale = height / heightOut
        pixelsOut = ipOut.flatten()
        i = 0
        threads = []
        for y in trange(heightOut):
            ys = y * yScale
            for x in range(widthOut):
                # see https://imagej.nih.gov/ij/developer/api/ij/ij/process/FloatProcessor.html#getInterpolatedPixel(double,double)
                # pixelsOut[i] = ip.getInterpolatedPixel(x * xScale, ys)
                t = threading.Thread(target=self.getInterpolatedPixel2, args=(x * xScale, ys, ip.flatten(), pixelsOut, i))
                threads.append(t)
                t.start()
                # pixelsOut[i] = self.getInterpolatedPixel2(x * xScale, ys, ip)
                i += 1
                if i == 10:
                    break
            if i == 10:
                break
        for t in threads:
            t.join()
        return np.reshape(pixelsOut, ip.shape)

    def prod1(self, m, v: np.ndarray):
        assert len(v.shape) == 1
        n = v.shape[0]
        out = np.zeros(n)
        for j in range(n):
            out[j] = 0.0
            for i in range(n):
                out[j] += m[j][i] * v[i]
        return out

    def prod2(self, a, b):
        out = np.zeros((len(a), len(b[0])))
        for i in range(len(a)):
            for j in range(len(b[i])):
                out[i][j] = 0.0
                for k in range(len(a[i])):
                    out[i][j] = out[i][j] + a[i][k] * b[k][j]
        return out

    def dx(self, ip):
        width, height = ip.shape

        pixels = ip
        outPixels = np.zeros_like(ip, dtype=float)

        for y in range(height):
            # Take forward/backward difference on edges.
            outPixels[y * width] = pixels[y * width + 1] - pixels[y * width]
            outPixels[y * width + width - 1] = pixels[y * width + width - 1] - pixels[y * width + width - 2]
            # Take central difference in interior.
            for x in range(1, width - 1):
                outPixels[y * width + x] = (pixels[y * width + x + 1] - pixels[y * width + x - 1]) * 0.5

        return outPixels

    def dy(self, ip):
        width, height = ip.shape

        pixels = ip
        outPixels = np.zeros_like(ip, dtype=float)

        # x
        for x in range(width):
            # Take forward / backward difference on edges.
            outPixels[x] = pixels[width + x] - pixels[x]
            outPixels[(height - 1) * width + x] = pixels[width * (height - 1) + x] - pixels[width * (height - 2) + x]

            # y
            for y in range(1, height - 1):
                # Take central difference in interior.
                outPixels[y * width + x] = (pixels[width * (y + 1) + x] - pixels[width * (y - 1) + x]) * 0.5

        return outPixels

    def dot(self, p1, p2):
        n = min(len(p1), len(p2))
        output = np.zeros(n, dtype=float)
        for i in range(n):
            output[i] = p1[i] * p2[i]
        return output

    def dotSum(self, p1, p2):
        sum = 0.0
        n = min(len(p1), len(p2))
        for i in range(n):
            sum += p1[i] * p2[i]
        return sum

    """
    Gaussian elimination (required by invert).

    This Python program is part of the book, "An Introduction to Computational
    Physics, 2nd Edition," written by Tao Pang and published by Cambridge
    University Press on January 19, 2006 written in Java.
    """

    @staticmethod
    def gaussian(a, index):
        n = len(index)
        c = np.zeros(n, dtype=float)

        # Initialize the index
        index = [i for i in range(n)]

        # Find the rescaling factors, one from each row
        for i in range(n):
            c1 = 0.0
            for j in range(n):
                c0 = abs(a[i][j])
                c1 = c0 if c0 > c1 else c1
            c[i] = c1

        # Search the pivoting element from each column
        k = 0
        for j in range(n - 1):
            pi1 = 0.0
            for i in range(j, n):
                pi0 = float(abs(a[index[i]][j]))
                pi0 /= c[index[i]]
                if pi0 > pi1:
                    pi1 = pi0
                    k = i

            # Interchange rows according to the pivoting order
            itmp = index[j]
            index[j] = index[k]
            index[k] = itmp
            for i in range(j + 1, n):
                pj = a[index[i]][j] / a[index[j]][j]
                # Record pivoting ratios below the diagonal
                a[index[i]][j] = pj
                # Modify other elements accordingly
                for l in range(j + 1, n):
                    a[index[i]][l] -= pj * a[index[j]][l]

    """
    Matrix inversion with the Gaussian elimination scheme.

    This Python program is part of the book, "An Introduction to Computational
    Physics, 2nd Edition," written by Tao Pang and published by Cambridge
    University Press on January 19, 2006 written in Java.
    """

    def invert(self, a):
        n = len(a)
        x = np.zeros((n, n), dtype=float)
        b = np.identity(n, dtype=float)
        index = np.zeros(n, dtype=int)

        # Transform the matrix into an upper triangle
        self.gaussian(a, index)

        # Update the matrix b[i][j] with the ratios stored
        for i in range(n - 1):
            for j in range(i + 1, n):
                for k in range(n):
                    b[index[j]][k] -= a[index[j]][i] * b[index[i]][k]

        # Perform backward substitutions
        for i in range(n):
            x[n - 1, i] = b[index[n - 1], i] / a[index[n - 1], n - 1]
            for j in range(n - 2, -1, -1):
                x[j, i] = b[index[j], i]
                for k in range(j + 1, n):
                    x[j, i] -= a[index[j], k] * x[k, i]
                x[j, i] /= a[index[j], j]
        return x

    @staticmethod
    def rootMeanSquare(ip):
        mean = 0.0
        pixels = ip.flatten()
        for i in range(0, len(pixels)):
            mean += pixels[i] * pixels[i]
        mean /= len(pixels)
        return sqrt(mean)

    def combine(self, ipOut, ip):
        pixels = ip.flatten()
        outPixels = ipOut.flatten()
        beta = 1.0 - self.alpha
        for i in range(0, len(pixels)):
            if pixels[i] != 0:
                outPixels[i] = self.alpha * outPixels[i] + beta * pixels[i]
        return np.reshape(outPixels, ip.shape)

    def subtract(self, ipOut, ip):
        pixels = ip.flatten()
        outPixels = ipOut.flatten()
        for i in range(0, len(pixels)):
            outPixels[i] = outPixels[i] - pixels[i]
        return np.reshape(outPixels, ipOut.shape)

    def warpAffine(self, ipOut, ip, wp):
        outPixels = ipOut.flatten()
        width, height = ipOut.shape
        p = 0
        for y in range(height):
            for x in range(width):
                xx = (1.0 + wp[0, 0]) * x + wp[0, 1] * y + wp[0, 2]
                yy = wp[1, 0] * x + (1.0 + wp[1, 1]) * y + wp[1, 2]
                outPixels[p] = self.getInterpolatedPixel2(xx, yy, ip)
                p += 1
        return np.reshape(outPixels, ipOut.shape)

    def warpTranslation(self, ipOut, ip, wp):
        outPixels = ipOut.flatten()
        width, height = ipOut.shape
        p = 0
        for y in range(height):
            for x in range(width):
                xx = x + wp[0, 0]
                yy = y + wp[1, 0]
                outPixels[p] = self.getInterpolatedPixel2(xx, yy, ip)  # TODO
                p += 1
        return np.reshape(outPixels, ipOut.shape)

    def getInterpolatedPixel1(self, x, y, ip, interpolationMethod='BILINEAR'):
        width, height = ip.shape
        if interpolationMethod == 'BICUBIC':
            pass
            # return getBicubicInterpolatedPixel(x, y)
        else:
            if x < 0.0:
                x = 0.0
            if x >= width - 1.0:
                x = width - 1.001
            if y < 0.0:
                y = 0.0
            if y >= height - 1.0:
                y = height - 1.001
            return self.getInterpolatedPixel2(x, y, ip)

    @lru_cache()
    def getInterpolatedPixel2(self, x, y, ip, dest=None, dest_idx=None):
        width, height = ip.shape
        xbase = int(x)
        ybase = int(y)
        xFraction = x - xbase
        yFraction = y - ybase
        offset = ybase * width + xbase
        lowerLeft = ip[offset]
        lowerRight = ip[offset + 1]
        upperRight = ip[offset + width + 1]
        upperLeft = ip[offset + width]
        # lowerLeft = ip[np.unravel_index(offset, ip.shape)]
        # lowerRight = ip[np.unravel_index(offset + 1, ip.shape)]
        # upperRight = ip[np.unravel_index(offset + width + 1, ip.shape)]
        # upperLeft = ip[np.unravel_index(offset + width, ip.shape)]
        if np.isnan(upperLeft) and xFraction >= 0.5:
            upperAverage = upperRight
        elif np.isnan(upperRight) and xFraction < 0.5:
            upperAverage = upperLeft
        else:
            upperAverage = upperLeft + xFraction * (upperRight - upperLeft)

        if np.isnan(lowerLeft) and xFraction >= 0.5:
            lowerAverage = lowerRight
        elif np.isnan(lowerRight) and xFraction < 0.5:
            lowerAverage = lowerLeft
        else:
            lowerAverage = lowerLeft + xFraction * (lowerRight - lowerLeft)
        if dest is None:
            if np.isnan(lowerAverage) and yFraction >= 0.5:
                return upperAverage
            elif np.isnan(upperAverage) and yFraction < 0.5:
                return lowerAverage
            else:
                return lowerAverage + yFraction * (upperAverage - lowerAverage)
        else:
            if np.isnan(lowerAverage) and yFraction >= 0.5:
                dest[dest_idx] = upperAverage
            elif np.isnan(upperAverage) and yFraction < 0.5:
                dest[dest_idx] = lowerAverage
            else:
                dest[dest_idx] = lowerAverage + yFraction * (upperAverage - lowerAverage)

