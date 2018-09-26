import codecs
import re
import numpy as np
import matplotlib.colors as cmx

def form_colormap(filepath, proportion=True, sep=False, spacing='c'):
    '''
    read text file containing colortable, and then form the colormap according to
    the argument given.
    sample text file structure: value, R, G, B

    Parameters
    ----------
    filepath: string
        The path of the colormap txt file.
    proportion: boolean, default is True
        When proportion is True, LinearSegmentedColormap will be formed
        by the input values.
    sep: boolean, default is False
        When sep is True, the colormap will be a set of color blocks 
        (no color gradient).
    spacing: string, default is 'c', only used when sep is False.
        When spacing is 'c', the color blocks will be equally spaced.
        A ListedColormap will be returned.
        When spacing is 'v', the length of color blocks will be based 
        on the input values. A LinearSegmentedColormap will be returned.
    '''
    inidict = {'red':None, 'green':None, 'blue':None}
    file_object = codecs.open(filepath, mode='r', encoding='GBK')
    all_the_text = file_object.read().strip()
    file_object.close()
    contents = re.split('[\s]+', all_the_text)
    nlin = int(len(contents) / 4)
    arr = np.array(contents)
    arr = arr.reshape(nlin, 4)
    if sep == True:
        if spacing == 'c':
            ar1 = arr.transpose()
            value = []
            count = 0
            while count < len(ar1[1]):
                value.append((int(ar1[1][count]) / 255, int(ar1[2][count]) / 255,
                              int(ar1[3][count]) / 255))
                count = count + 1
            return cmx.ListedColormap(value, 256)
    elif sep == False:
        value = []
        r = []
        g = []
        b = []
        for i in arr:
            value.append(float(i[0]))
            r.append(int(i[1]) / 255)
            g.append(int(i[2]) / 255)
            b.append(int(i[3]) / 255)
            if len(value) > 1:
                if value[-1] < value[-2]:
                    raise ValueError('Values must be in order')
        inivalue = value[0]
        maxvalue = value[-1]
        drange = maxvalue-inivalue
        rpart = []
        gpart = []
        bpart = []
        if proportion == True:
            count = 0
            while count < len(value):
                tupr = ((value[count] - inivalue) / drange, r[count], r[count])
                tupg = ((value[count] - inivalue) / drange, g[count], g[count])
                tupb = ((value[count] - inivalue) / drange, b[count], b[count])
                rpart.append(tupr)
                gpart.append(tupg)
                bpart.append(tupb)
                count=count + 1
        elif proportion == False:
            count = 0
            while count < len(value):
                tupr = (count / len(value), r[count], r[count])
                tupg = (count / len(value), g[count], g[count])
                tupb = (count / len(value), b[count], b[count])
                rpart.append(tupr)
                gpart.append(tupg)
                bpart.append(tupb)
                count = count + 1
        inidict['red'] = rpart
        inidict['green'] = gpart
        inidict['blue'] = bpart
        return cmx.LinearSegmentedColormap('my_colormap', inidict, 256)
