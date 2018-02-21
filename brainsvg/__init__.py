import numpy as np
import scipy
from scipy import io
import nibabel as nib

def load_mouse():
    mouse = scipy.io.loadmat('mouse/MouseAllenCortex.mat')
    namesOrig = mouse['IsoCortexNames']

    import networkx as nx
    G = nx.read_graphml('mouse/mouse_brain_1.graphml')
    Ap = nx.to_numpy_matrix(G, weight='pvalue_ipsi_weight')
    Aw = nx.to_numpy_matrix(G, weight='w_ipsi_weight')

    names = []
    for x in xrange(0,len(G.node)):
        names.append(str(G.node[G.nodes()[x]]['name']))

    idx = []
    names_new = []
    for x in xrange(0,len(names)):
        if [s for s in namesOrig if names[x] in s]:
            idx.append(x)
            names_new.append(names[x])

    idx = np.array(idx)
    labels = np.array(names_new).copy()
    del names_new

    mat = np.squeeze(Ap[idx[:,None],idx[None,:]])
    mat_thr = mat.copy()
    thresh = 0.05
    Atmp = Aw[idx[:,None],idx[None,:]]

    mat_thr[np.where((mat <= thresh) & (mat != 0))] = 1
    mat_thr[(mat > thresh)] = 0
    mat = np.concatenate((mat_thr.copy(), mat_thr.copy().T), axis=1)

    labels_color = np.genfromtxt('mouse/mouse_terms_new_labels.csv', dtype=str, delimiter=',')[:,1]

    return mat, labels, labels_color

def load_rat(thresh=2.):

    mat = np.genfromtxt('rat/rat_matrix_2013_cut.csv', dtype=int, delimiter=',')
    names = np.genfromtxt('rat/rat_matrix_2013_labels_cut.csv', dtype=str, delimiter=',')
    labels_color = np.genfromtxt('rat/rat_label_colors_cut.csv', dtype=str, delimiter=',')[:,1]
    mat = mat.copy()[:,np.concatenate((range(7),range(19,len(names)+12)))]

    mat_thr = mat.copy()
    mat_thr[np.where(mat < thresh)] = 0
    mat_thr[np.where(mat >= thresh)] = 1
    mat = np.concatenate((mat_thr.copy(), mat_thr.copy().T), axis=1)
    del mat_thr

    labels = names.copy()

    return mat, labels, labels_color

def load_human():

    emat = scipy.io.loadmat('../templates/human/economo_data.mat')
    labels = []
    for x in emat['ECONOMO_regions'][0]:
        labels.append(x[0].encode('ascii'))


    emat = scipy.io.loadmat('../templates/human/economo_data.mat')
    labels = []
    for x in emat['ECONOMO_regions'][0]:
        labels.append(x[0].encode('ascii'))

    data = []
    header = []
    thresh = 100 # currently arbitrarily high, but could be used to remove data columns with excessive NaNs...
    for x in emat.keys():
        if str(x).startswith('layer'):# or str(x).startswith('total'):
            # if '_thickness_dome' in str(x):
            if np.sum(np.isnan(emat[x])) < thresh:
                data.append(emat[x])
                header.append(x)
    data = np.array(data).squeeze().T

    return data, header, labels


def parse_xml(species, filename, order1, cb='jet', split=None):

    import xml.etree.ElementTree as ET
    from colormap import rgb2hex
    import pandas as pd
    import matplotlib

    path_style = "stroke:#000000;stroke-width:0;stroke-opacity:0;fill-opacity:1;fill:"
    SVG_NS = "http://www.w3.org/2000/svg"

    if split:
        tree = ET.parse('../templates/' + species + '/' + species + '_' + split + '_template.svg')
    else:
        tree = ET.parse('../templates/' + species + '/' + species + '_template.svg')
    root = tree.getroot()

    sp = list(pd.read_csv('../templates/' + species + '/' + species + '_terms.csv', header=-1)[0])

    if cb == 'jet':
        cmap = plt.cm.get_cmap('jet')
    if cb == 'archi':
        colorsList = np.vstack((np.linspace(0.5, 1.0, 50), np.linspace(0.0, 1.0, 50), np.linspace(0.5, 1.0, 50))).T
        cmap = matplotlib.colors.ListedColormap(colorsList)
    if cb == 'paleo':
        colorsList = np.vstack((np.linspace(0.0,1.0,50), np.linspace(0.5, 1.0, 50), np.linspace(0.5, 1.0, 50))).T
        cmap = matplotlib.colors.ListedColormap(colorsList)
    if cb == 'wta':
        colorsList = [(0.5, 0, 0.5),(1.0, 1.0, 1.0),(0,0.5,0.5)]
        cmap = matplotlib.colors.ListedColormap(colorsList)

    normalized = (order1-min(order1))/(max(order1)-min(order1))
    col = cmap(normalized)

    for node in root.findall('.//{%s}path' % SVG_NS):
        matching = []
        if species == 'cat':
            matching = [n for n, s in enumerate(sp) if
                        node.attrib['id'] == str('path' + s) or
                        node.attrib['id'] == str('path' + s + '2') or
                        node.attrib['id'] == str('path' + s + '_2')]
        elif species == 'macaque':
            matching = [n for n, s in enumerate(sp) if
                        node.attrib['id'] == str(s) or
                        node.attrib['id'] == str(s + '-1') or
                        node.attrib['id'] == str(s + '-2')]
        elif species == 'human':
            matching = [n for n, s in enumerate(sp) if
                    node.attrib['fill'] == str(s)]
        else:
            if 'id' in node.keys():
                matching = [n for n, s in enumerate(sp) if
                        node.attrib['id'] == str(s) or
                        node.attrib['id'] == (str(s) + '1') or
                        node.attrib['id'] == (str(s) + '2')]

        if len(matching) > 0:
            a = tuple(list(col[matching[0]]))
            node.attrib['style'] = path_style + rgb2hex(a[0],a[1],a[2],a[3]) + ';'
            node.attrib['fill'] = ""

    tree.write(filename)

def run_svg_results(species,paint,labels_dict,vertices,triangles,pir_regions,hip_regions):

    import gdist
    from scipy import stats

    pir = np.array(np.where(np.in1d(paint,pir_regions))[0], dtype=np.int32)
    hip = np.array(np.where(np.in1d(paint,hip_regions))[0], dtype=np.int32)

    data_dist_pir = gdist.compute_gdist(vertices, triangles, source_indices=pir)
    data_dist_hip = gdist.compute_gdist(vertices, triangles, source_indices=hip)

    dist_labels_pir = []
    dist_labels_hip = []
    ks_stat = []
    for i in labels_dict:
        labs = np.where(paint == i)[0]
        dist_labels_pir.append(np.mean(data_dist_pir[labs]))
        dist_labels_hip.append(np.mean(data_dist_hip[labs]))
        ks_raw, p = stats.ks_2samp(data_dist_pir[labs], data_dist_hip[labs])
        ks_stat.append(ks_raw)

    dist_labels = np.min(np.vstack((dist_labels_pir, dist_labels_hip)), axis = 0)

    lab_pir = np.where(np.argmin(np.vstack((dist_labels_pir, dist_labels_hip)), axis = 0) == 0)[0]
    lab_hip = np.where(np.argmin(np.vstack((dist_labels_pir, dist_labels_hip)), axis = 0) == 1)[0]

    #dist_labels[lab_pir] = ((dist_labels[lab_pir] / np.max(dist_labels[lab_pir])) * -1 ) + 1
    #dist_labels[lab_hip] = (dist_labels[lab_hip] / np.max(dist_labels[lab_hip])) * -1

    # for wta labeling
    dist_labels[lab_pir] = 1
    dist_labels[lab_hip] = -1

    parse_xml(species, './fig.dist.%s.hip.svg' % species, np.array(dist_labels_hip), 'archi')
    parse_xml(species, './fig.dist.%s.pir.svg' % species, np.array(dist_labels_pir), 'paleo')
    parse_xml(species, './fig.dist.%s.wta.svg' % species, np.array(dist_labels), 'wta')

def run_mouse():

    mousedir = '../templates/mouse/'

    #mat, labels, labels_color = ns.load_mouse()
    labels_dict = np.genfromtxt(mousedir + 'mouse_terms_dictionary.csv', dtype=int,delimiter='\n')

    f = open(mousedir + "mouse.paint","r")
    for i in range(113):
        f.readline()
    paint = []
    for line in f:
        paint.append(np.int(line.strip('\n').split()[3]))
    paint = np.array(paint)

    mouse_surf = nib.load(mousedir + 'mouse.L.gii')
    vertices = np.array(mouse_surf.darrays[0].data, dtype=np.float64)
    triangles = np.array(mouse_surf.darrays[1].data, dtype=np.int32)


    s2 = np.where(paint == 50)[0]
    for i in s2:
        if vertices[i][0] > -2.0:
            paint[i] = 0

    run_svg_results('mouse',paint,labels_dict,vertices,triangles,[38],[17,18,19,20,21,22])

def run_rat():

    ratdir = '../templates/rat/'

    labels_dict = np.genfromtxt(ratdir + 'rat_terms_dictionary.csv', dtype=int,delimiter='\n')

    f = open(ratdir + "rat.paint","r")
    for i in range(108):
        f.readline()
    paint = []
    for line in f:
        paint.append(np.int(line.strip('\n').split()[3]))
    paint = np.array(paint)

    rat_surf = nib.load(ratdir + '/rat.L.gii')
    vertices = np.array(rat_surf.darrays[0].data, dtype=np.float64)
    triangles = np.array(rat_surf.darrays[1].data, dtype=np.int32)

    s2 = np.where(paint == 49)[0]
    for i in s2:
        if vertices[i][0] > -3.0:
            paint[i] = 0

    run_svg_results('rat',paint,labels_dict,vertices,triangles,[60],[11,12,13,14,15,17])

def run_macaque():

    species_dir = '../templates/macaque/'

    macaque_surf = nib.load(species_dir + 'MacaqueYerkes19.L.midthickness.32k_fs_LR.surf.gii')
    vertices = np.array(macaque_surf.darrays[0].data, dtype=np.float64)
    triangles = np.array(macaque_surf.darrays[1].data, dtype=np.int32)

    labels = np.genfromtxt(species_dir + 'MarkovCC12_M132_29-injected-areas.32k_fs_LR.txt', dtype=str, delimiter='\n')

    label_indices = np.array(nib.load(species_dir + 'MarkovCC12_M132_29-injected-areas.32k_fs_LR.dlabel.nii').get_header().get_index_map(1)[0].vertex_indices)
    parcels_indices = np.array(nib.load(species_dir + 'MarkovCC12_M132_91-area.32k_fs_LR.dlabel.nii').get_header().get_index_map(1)[0].vertex_indices)
    paint = nib.load(species_dir + 'MarkovCC12_M132_29-injected-areas.32k_fs_LR.dlabel.nii').get_data().squeeze()[range(len(label_indices))]
    parcels = nib.load(species_dir + 'MarkovCC12_M132_91-area.32k_fs_LR.dlabel.nii').get_data().squeeze()[range(len(parcels_indices))]

    medial = nib.load(species_dir + 'Macaque.MedialWall.32k_fs_LR.dlabel.nii').get_data().squeeze()[range(32492)]
    wall = np.where(medial == 1)[0]
    cortex = np.where(medial != 1)[0]
    labels_dict = labels

    import gdist
    from surfdist import surfdist, utils

    surf = []
    surf.append(vertices)
    surf.append(triangles)
    vertices, triangles = utils.surf_keep_cortex(surf,cortex)

    pir = np.array(parcels_indices[np.where(np.in1d(parcels,[58,57]))[0]], dtype=np.int32) # Piriform, INSULA, OPRO , 48, 57
    hip = np.array(parcels_indices[np.where(np.in1d(parcels,[16,20]))[0]], dtype=np.int32) # 71,  SUBICULUM, 24a, 29/30, 16, 20,
    #,57,65,
    data_dist_pir = utils.recort(gdist.compute_gdist(vertices, triangles, source_indices=utils.translate_src(pir, cortex)), surf, cortex)
    data_dist_hip = utils.recort(gdist.compute_gdist(vertices, triangles, source_indices=utils.translate_src(hip, cortex)), surf, cortex)

    dist_labels_pir = []
    dist_labels_hip = []
    for i in np.unique(paint):
        if i > 0.0:
            labs = np.where(paint == i)[0]
            # set values for mask as mean, median, or min
            dist_labels_pir.append(np.mean(data_dist_pir[parcels_indices[labs]]))
            dist_labels_hip.append(np.mean(data_dist_hip[parcels_indices[labs]]))

    dist_labels = np.min(np.vstack((dist_labels_pir, dist_labels_hip)), axis = 0)

    lab_pir = np.where(np.argmin(np.vstack((dist_labels_pir, dist_labels_hip)), axis = 0) == 0)[0]
    lab_hip = np.where(np.argmin(np.vstack((dist_labels_pir, dist_labels_hip)), axis = 0) == 1)[0]

    #dist_labels[lab_pir] = ((dist_labels[lab_pir] / np.max(dist_labels[lab_pir])) * -1 ) + 1
    #dist_labels[lab_hip] = (dist_labels[lab_hip] / np.max(dist_labels[lab_hip])) * -1

    # for binary labeling
    dist_labels[lab_pir] = 1
    dist_labels[lab_hip] = -1

    parse_xml('macaque', './fig.dist.macaque.hip.svg', np.array(dist_labels_hip), cb='archi')
    parse_xml('macaque', './fig.dist.macaque.pir.svg', np.array(dist_labels_pir), cb='paleo')
    parse_xml('macaque', './fig.dist.macaque.wta.svg', np.array(dist_labels), cb='wta')


def run_human():

    data, header, labels = load_human()
    species = 'human'
    species_dir = '../templates/' + species + '/'

    human_surf = nib.load(species_dir + 'S900.L.midthickness_MSMAll.32k_fs_LR.surf.gii')
    vertices = np.array(human_surf.darrays[0].data, dtype=np.float64)
    triangles = np.array(human_surf.darrays[1].data, dtype=np.int32)

    label_indices = np.array(nib.load(species_dir + 'economo.dlabel.nii').get_header().get_index_map(1)[0].vertex_indices)
    paint = nib.load(species_dir + 'economo.dlabel.nii').get_data().squeeze()[range(len(label_indices))]

    medial = nib.load(species_dir + 'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii').get_data().squeeze()[range(32492)]
    wall = np.where(medial == 1)[0]
    cortex = np.where(medial != 1)[0]

    import gdist
    from surfdist import surfdist, utils

    surf = []
    surf.append(vertices)
    surf.append(triangles)
    vertices, triangles = utils.surf_keep_cortex(surf,cortex)

    pir = np.array(label_indices[np.where(np.in1d(paint,[13]))[0]], dtype=np.int32) # PARAINSULA 58, 24a, #16,
    hip = np.array(label_indices[np.where(np.in1d(paint,[15,21]))[0]], dtype=np.int32) # SUBICULUM, ProM # , 65
    #,57,65,
    data_dist_pir = utils.recort(gdist.compute_gdist(vertices, triangles, source_indices=utils.translate_src(pir, cortex)), surf, cortex)
    data_dist_hip = utils.recort(gdist.compute_gdist(vertices, triangles, source_indices=utils.translate_src(hip, cortex)), surf, cortex)

    dist_labels_pir = []
    dist_labels_hip = []
    label_dict = np.genfromtxt(species_dir + 'economo_dictionary.csv', dtype=int, delimiter='\n')
    for i in label_dict:
        #if i > 0.0:
        labs = np.where(paint == i)[0]
        # set values for mask as mean, median, or min
        dist_labels_pir.append(np.mean(data_dist_pir[label_indices[labs]]))
        dist_labels_hip.append(np.mean(data_dist_hip[label_indices[labs]]))

    dist_labels = np.min(np.vstack((dist_labels_pir, dist_labels_hip)), axis = 0)

    lab_pir = np.where(np.argmin(np.vstack((dist_labels_pir, dist_labels_hip)), axis = 0) == 0)[0]
    lab_hip = np.where(np.argmin(np.vstack((dist_labels_pir, dist_labels_hip)), axis = 0) == 1)[0]

    dist_labels[lab_pir] = ((dist_labels[lab_pir] / np.max(dist_labels[lab_pir])) * -1 ) + 1
    dist_labels[lab_hip] = (dist_labels[lab_hip] / np.max(dist_labels[lab_hip])) * -1

    # for binary labeling
    dist_labels[lab_pir] = 1
    dist_labels[lab_hip] = -1

    def reduce_human(d_in):
        return np.concatenate((d_in[0:17],[np.mean(d_in[17:19])], d_in[19::]))

    for side in ['lateral', 'medial']:
        parse_xml('human', './fig.dist.%s.%s.hip.svg' % (species, side), reduce_human(dist_labels_hip), cb='archi', split=side)
        parse_xml('human', './fig.dist.%s.%s.pir.svg' % (species, side), reduce_human(dist_labels_pir), cb='paleo', split=side)
        parse_xml('human', './fig.dist.%s.%s.wta.svg' % (species, side), reduce_human(dist_labels), cb='wta', split=side)
