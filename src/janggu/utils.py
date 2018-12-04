"""Utilities for janggu """

import json
import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from HTSeq import BED_Reader
from HTSeq import GFF_Reader
from HTSeq import GenomicFeature
from HTSeq import GenomicInterval

try:
    import matplotlib.pyplot as plt  # pylint: disable=import-error
except ImportError:  # pragma: no cover
    plt = None

try:
    import pyBigWig
except ImportError:  # pragma: no cover
    pyBigWig = None
try:
    import seaborn as sns  # pylint: disable=import-error
except ImportError:  # pragma: no cover
    sns = None

try:
    from sklearn.manifold import TSNE  # pylint: disable=import-error
except ImportError:  # pragma: no cover
    TSNE = None

try:
    from urllib import urlcleanup, urlretrieve
except ImportError:
    try:
        from urllib.request import urlcleanup, urlretrieve
    except ImportError as exception:  # pragma: no cover
        urlretrieve = None
        urlcleanup = None


def _get_output_root_directory():
    """Function returns the output root directory."""
    if "JANGGU_OUTPUT" not in os.environ:  # pragma: no cover

        os.environ['JANGGU_OUTPUT'] = os.path.join(os.path.expanduser("~"),
                                                   'janggu_results')
        print('environment variable JANGGU_OUTPUT not set.'
              ' Will use JANGGU_OUTPUT={}'.format(os.environ['JANGGU_OUTPUT']))
    return os.environ['JANGGU_OUTPUT']


def _get_output_data_location(datatags):
    """Function returns the output location for the dataset.

    Parameters
    ------------
    datatags : list(str)
        Tags describing the dataset. E.g. ['testdna'].
    """
    outputdir = _get_output_root_directory()

    args = (outputdir, 'datasets')
    if datatags is not None:
        args += tuple(datatags)

    return os.path.join(*args)


def sequences_from_fasta(fasta, string='dna'):
    """Obtains nucleotide or peptide sequences from a fasta file.

    Parameters
    -----------
    fasta : str
        Fasta-filename
    string : str
        Either dna or protein.

    Returns
        List of Biostring sequences
    """

    file_ = open(fasta)
    if string == 'dna':
        alpha = IUPAC.unambiguous_dna
    else:
        alpha = IUPAC.protein
    gen = SeqIO.parse(file_, "fasta", alpha)
    seqs = [item for item in gen]

    return seqs


LETTERMAP = {k: i for i, k in enumerate(sorted(IUPAC.unambiguous_dna.letters))}
NNUC = len(IUPAC.unambiguous_dna.letters)

# mapping of nucleotides to integers
NMAP = defaultdict(lambda: -1024)
NMAP.update(LETTERMAP)

# mapping of amino acids to integers
LETTERMAP = {k: i for i, k in enumerate(sorted(IUPAC.protein.letters))}
PMAP = defaultdict(lambda: -1024)
PMAP.update(LETTERMAP)


def seq2ind(seq):
    """Transforms a biological sequence into an int array.

    Each nucleotide or amino acid maps to an integer between
    zero and len(alphabet) - 1.

    Any other characters (e.g. 'N') are represented by a negative value
    to avoid confusion with valid nucleotides.

    Parameters
    ----------
    seq : str, Bio.SeqRecord or Bio.Seq.Seq
        Sequence represented as string, SeqRecord or Seq.

    Returns
    -------
    list(int)
        Integer array representation of the biological sequence.
    """

    if isinstance(seq, SeqRecord):
        seq = seq.seq
    if isinstance(seq, (str, Seq)):
        if type(seq.alphabet) is type(IUPAC.unambiguous_dna):
            return [NMAP[x.upper()] for x in seq]
        elif type(seq.alphabet) is type(IUPAC.protein):
            return [PMAP[x.upper()] for x in seq]
    raise TypeError('seq2ind: Format is not supported')


def sequence_padding(seqs, length):
    """This function truncates or pads the sequences
    to achieve fixed length sequences.

    Padding of the sequence is achieved by appending '.'

    Parameters
    ----------
    seqs : str, Bio.SeqRecord or Bio.Seq.Seq
        Sequence represented as string, SeqRecord or Seq.

    Returns
    -------
    list(Bio.SeqRecord)
        Padded sequence represented as string, SeqRecord or Seq.
    """
    seqs_ = deepcopy(seqs)
    for idx, seq in enumerate(seqs_):
        if len(seq) < length:
            seqs_[idx] += '.' * (length - len(seq))
        else:
            seqs_[idx] = seq[:length]
    return seqs_


def as_onehot(iseq, order, alphabetsize):
    """Converts a index sequence into one-hot representation.

    This method is used to transform a biological sequence
    for a given batch, represented by integer indices,
    into a one-hot representation.

    Parameters
    ----------
    iseq: numpy.array
        Array that holds the indices for a given batch.
        The dimensions of the array correspond to
        `(batch_size, sequence_length + 2*flank - order + 1)`.
    order: int
        Order of the sequence representation. Used for higher-order
        motif modelling.

    Returns
    -------
    numpy.array
        One-hot representation of the batch. The dimension
        of the array is given by
        `(batch_size, sequence length, 1, pow(alphabetsize, order))`
    """

    onehot = np.zeros((len(iseq),
                       iseq.shape[1], 1,
                       pow(alphabetsize, order)), dtype='int8')
    for nuc in np.arange(pow(alphabetsize, order)):
        onehot[:, :, 0, nuc][iseq == nuc] = 1

    return onehot


def _complement_index(idx, order):
    rev_idx = np.arange(NNUC)[::-1]
    irc = 0
    for iord in range(order):
        nuc = idx % NNUC
        idx = idx // NNUC
        irc += rev_idx[nuc] * pow(NNUC, order - iord - 1)

    return irc


def complement_permmatrix(order):
    """This function returns a permutation matrix for computing
    the complementary DNA strand one-hot representation for a given order.

    Parameters
    ----------
    order : int
        Order of the one-hot representation

    Returns
    -------
    np.array
        Permutation matrix
    """
    perm = np.zeros((pow(NNUC, order), pow(NNUC, order)))
    for idx in range(pow(NNUC, order)):
        jdx = _complement_index(idx, order)
        perm[jdx, idx] = 1
    return perm


def get_genome_size(refgenome='hg19', outputdir='./', skip_random=True):
    """Get genome size.

    This function loads the genome size for a specified reference genome
    into a dict. The reference genome sizes are obtained from
    UCSC genome browser.

    Parameters
    ----------
    refgenome : str
        Reference genome name. Default: 'hg19'.
    outputdir : str
        Directory in which the downloaded *.chrom.sizes file will be stored.
    skipRandom : True

    Returns
    -------
    dict()
        Dictionary with chromosome names as keys and their respective lengths
        as values.
    """
    if urlcleanup is None:  # pragma: no cover
        raise Exception('urllib not available. Please install urllib3.')

    outputfile = os.path.join(outputdir, '{}.chrom.sizes'.format(refgenome))
    if not os.path.exists(outputfile):  # pragma: no cover
        # not part of unit tests, because this requires internet connection
        urlpath = 'http://hgdownload.cse.ucsc.edu/goldenPath/{}/bigZips/{}.chrom.sizes'.format(
            refgenome, refgenome)

        # From the md5sum.txt we extract the
        print("Downloading {}".format(urlpath))
        urlcleanup()
        urlretrieve(urlpath.format(refgenome), outputfile)

    content = pd.read_csv(outputfile, sep='\t', names=['chr', 'length'],
                          index_col='chr')
    if skip_random:
        fltr = [True if len(name.split('_')) <= 1 else False for name in content.index]
        content = content[fltr]
    return content.to_dict()['length']


def _get_genomic_reader(filename):
    """regions from a BED_Reader or GFF_Reader.
    """
    if isinstance(filename, str) and filename.endswith('.bed'):
        regions_ = BED_Reader(filename)
    elif isinstance(filename, str) and (filename.endswith('.gff') or
                                        filename.endswith('.gtf')):
        regions_ = GFF_Reader(filename)
    else:
        raise Exception('Regions must be a bed, gff or gtf-file.')

    return regions_


def _iv_to_str(chrom, start, end):
    """Converts a genomic interval into a string representation 'chr:start-end'."""
    return '{}:{}-{}'.format(chrom, start, end)


def _str_to_iv(givstr, template_extension):
    """Converts a string representation 'chr:start-end' into genomic coordinates."""
    sub = givstr.split(':')
    if len(sub) == 1:
        return (sub[0], )

    chr_ = sub[0]
    start = int(sub[1].split('-')[0])
    end = int(sub[1].split('-')[1])
    return (chr_, start - template_extension, end + template_extension)


def get_chrom_length(length, resolution):
    """obtain the chromosome length for a given resolution."""
    return int(np.ceil(length/resolution))


def get_genome_size_from_regions(regions):
    """Get genome size.

    This function loads the genome size for a specified reference genome
    into a dict. The reference genome sizes are obtained from
    UCSC genome browser.

    Parameters
    ----------
    regions : str or GenomicIndexer
        Either a path pointing to a BED or GFF file containing genomic regions
        or a GenomicIndexer object.

    Returns
    -------
    dict()
        Dictionary with chromosome names as keys and their respective lengths
        as values.
    """

    regions_ = regions
    if isinstance(regions, str):
        regions_ = _get_genomic_reader(regions)

    gsize = {}
    for region in regions_:
        if isinstance(region, GenomicFeature):
            iv = region.iv
        elif isinstance(region, GenomicInterval):
            iv = region
        if iv.chrom not in gsize:
            gsize[iv.chrom] = iv.end
        elif gsize[iv.chrom] < iv.end:
            gsize[iv.chrom] = iv.end
    return gsize


def export_json(output_dir, name, results, filesuffix='json',
                annot=None, row_names=None):
    """Method that dumps the results in a json file.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    filesuffix : str
        Target file ending. Default: 'json'.
    annot: None, dict
        Annotation data. If encoded as dict the key indicates the name,
        while the values holds a list of annotation labels. Default: None.
    row_names : None or list
        List of row names. Default: None.
    """

    filename = os.path.join(output_dir, name + '.' + filesuffix)

    content = {}  # needed for py27
    with open(filename, 'w') as jsonfile:
        try:
            content.update({'-'.join(key): results[key]['value'].tolist() for key in results})
        except AttributeError:
            content.update({'-'.join(key): results[key]['value'] for key in results})
        if annot is not None:
            content.update({'annot': annot})

        if row_names is not None:
            content.update({'row_names': row_names})
        json.dump(content, jsonfile)


def export_tsv(output_dir, name, results, filesuffix='tsv', annot=None, row_names=None):
    """Method that dumps the results as tsv file.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    filesuffix : str
        File ending. Default: 'tsv'.
    annot: None, dict
        Annotation data. If encoded as dict the key indicates the name,
        while the values holds a list of annotation labels. For example,
        this can be used to store the true output labels.
        Default: None.
    row_names : None, list
        List of row names. For example, chromosomal loci. Default: None.
    """

    filename = os.path.join(output_dir, name + '.' + filesuffix)
    try:
        # check if the result is iterable
        iter(results[list(results.keys())[0]]['value'])
        _rs = {'-'.join(k): results[k]['value'] for k in results}
    except TypeError:
        # if the result is not iterable, wrap it up as list
        _rs = {'-'.join(k): [results[k]['value']] for k in results}
    _df = pd.DataFrame.from_dict(_rs)
    for _an in annot or []:
        _df['annot.' + _an] = annot[_an]
    if row_names is not None:
        _df['row_names'] = row_names
    _df.to_csv(filename, sep='\t', index=False)


def export_plotly(output_dir, name, results, annot=None, row_names=None):
    """This method exports data for interactive visualization.

    Essentially, it exports a table with the filename suffix
    being '.ply'. In contrast table files generally, which have
    the file ending '.tsv', .ply file is ment to have a more specific
    structure so that the Dash app can interpret it.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    annot: None, dict
        Annotation data. If encoded as dict the key indicates the name,
        while the values holds a list of annotation labels.
        For example, this can be used to store color information to overlay
        with the interactive scatter plot. Default: None.
    row_names : None, list
        List of row names. For instance, chromosomal loci. Default: None.
    """
    # this produces a normal json file, but for the dedicated
    # purpose of visualization in the dash app.
    export_tsv(output_dir, name, results, 'ply', annot, row_names)


def export_score_plot(output_dir, name, results, figsize=None,  # pylint: disable=too-many-locals
                      xlabel=None, ylabel=None, fform=None):
    """Method that dumps the results in a json file.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    figsize : tuple(int, int)
        Used to specify the figure size for matplotlib.
    xlabel : str or None
        xlabel used for the plot.
    ylabel : str or None
        ylabel used for the plot.
    fform : str or None
        Output file format. E.g. 'png', 'eps', etc. Default: 'png'.
    """
    if plt is None:  # pragma: no cover
        raise Exception('matplotlib not available. Please install matplotlib.')

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    ax_ = fig.add_axes([0.1, 0.1, .55, .5])

    ax_.set_title(name)
    for keys in results:
        # avg might be returned using a custom function
        x_score, y_score, auxstr = results[keys]['value']
        label = "{}".format('-'.join([keys[0][:8], keys[1], keys[2]]))
        if isinstance(auxstr, str):
            label += ' ' + auxstr
        ax_.plot(x_score, y_score, label=label)

    lgd = ax_.legend(bbox_to_anchor=(1.05, 1),
                     loc=2, prop={'size': 10}, ncol=1)
    if xlabel is not None:
        ax_.set_xlabel(xlabel, size=14)
    if ylabel is not None:
        ax_.set_ylabel(ylabel, size=14)
    if fform is not None:
        fform = fform
    else:
        fform = 'png'
    filename = os.path.join(output_dir, name + '.' + fform)
    fig.savefig(filename, format=fform,
                dpi=1000,
                bbox_extra_artists=(lgd,), bbox_inches="tight")


def export_bigwig(output_dir, name, results, gindexer=None):
    """Export predictions to bigwig.

    This function exports the predictions to bigwig format which allows you to
    inspect the predictions in a genome browser.
    Importantly, gindexer must contain non-overlapping windows!

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    gindexer : GenomicIndexer
        GenomicIndexer that links the prediction for a certain region to
        its associated genomic coordinates.
    """

    if pyBigWig is None:  # pragma: no cover
        raise Exception('pyBigWig not available. '
                        '`export_bigwig` requires pyBigWig to be installed.')

    if gindexer is None:
        raise ValueError('Please specify a GenomicIndexer for export_to_bigiwig')

    genomesize = {}

    # extract genome size from gindexer
    # check also if sorted and non-overlapping
    for region in gindexer:
        if region.chrom not in genomesize:
            genomesize[region.chrom] = region.end
        if genomesize[region.chrom] < region.end:
            genomesize[region.chrom] = region.end

    bw_header = [(chrom, genomesize[chrom])
                 for chrom in genomesize]

    # the last dimension holds the conditions. Each condition
    # needs to be stored in a separate file
    for modelname, layername, condition in results:
        bw_file = pyBigWig.open(os.path.join(
            output_dir,
            '{prefix}.{model}.{output}.{condition}.bigwig'.format(
                prefix=name, model=modelname,
                output=layername, condition=condition)), 'w')
        bw_file.addHeader(bw_header)
        pred = results[modelname, layername, condition]['value']
        # compute the ratio between binsize and stepsize
        bsss = float(gindexer.binsize) / float(gindexer.stepsize)
        if bsss < 1.:
            bsss = 1.
        ppi = int(np.rint(len(pred)/(len(gindexer) - 1. + bsss)))

        # case 1) stepsize >= binsize
        # then bsss = 1; ppi = len(pred)/len(gindexer)
        #
        # case 2) stepsize < binsize
        # then bsss > 1; ppi = len(pred)/ (len(gindexer) -1 + bsss)
        resolution = int(region.length / bsss) // ppi
        for idx, region in enumerate(gindexer):

            val = [float(p) for p in pred[(idx*ppi):((idx+1)*ppi)]]

            bw_file.addEntries(str(region.chrom),
                               int(region.start),
                               values=val,
                               span=int(resolution),
                               step=int(resolution))
        bw_file.close()


def export_bed(output_dir, name, results,  # pylint: disable=too-many-locals
               gindexer=None, resolution=None):
    """Export predictions to bed.

    This function exports the predictions to bed format which allows you to
    inspect the predictions in a genome browser.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    gindexer : GenomicIndexer
        GenomicIndexer that links the prediction for a certain region to
        its associated genomic coordinates.
    resolution : int
        Used to output the results.
    """

    if gindexer is None:
        raise ValueError('Please specify a GenomicIndexer for export_to_bed')
    if resolution is None:
        raise ValueError('Resolution must be specify')
    # the last dimension holds the conditions. Each condition
    # needs to be stored in a separate file

    for modelname, layername, condition in results:
        bed_content = pd.DataFrame(columns=['chr', 'start',
                                            'end', 'name', 'score'])
        for ridx, region in enumerate(gindexer):
            pred = results[modelname, layername, condition]['value']

            nsplit = (region.end-region.start)//resolution

            starts = list(range(region.start,
                                region.end,
                                resolution))
            ends = list(range(region.start + resolution,
                              region.end + resolution,
                              resolution))
            cont = {'chr': [region.chrom] * nsplit,
                    'start': [s for s in starts],
                    'end': [e for e in ends],
                    'name': ['.'] * nsplit,
                    'score': pred[ridx*nsplit:((ridx+1)*nsplit)]}

            bed_entry = pd.DataFrame(cont)
            bed_content = bed_content.append(bed_entry, ignore_index=True, sort=False)

        bed_content.to_csv(os.path.join(
            output_dir,
            '{prefix}.{model}.{output}.{condition}.bed'.format(
                prefix=name, model=modelname,
                output=layername, condition=condition)),
                           sep='\t', header=False, index=False,
                           columns=['chr', 'start', 'end', 'name', 'score'])


def export_clustermap(output_dir, name, results, fform=None, figsize=None,  # pylint: disable=too-many-locals
                      annot=None,
                      method='ward', metric='euclidean', z_score=None,
                      standard_scale=None, row_cluster=True, col_cluster=True,
                      row_linkage=None, col_linkage=None, row_colors=None,
                      col_colors=None, mask=None, **kwargs):
    """Create of clustermap of the feature activities.

    This method utilizes
    `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_
    to illustrate feature activities of the neural net.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    fform : str or None
        Output file format. E.g. 'png', 'eps', etc. Default: 'png'.
    figsize : tuple(int, int) or None
        Used to specify the figure size for matplotlib.
    annot : None
        Row annotation is used to create a row color annotation track
        for the figure.
    method : str
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
        Default: 'ward'
    metric : str
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
        Default: 'euclidean'
    z_score : int or None
        Whether to transform rows or columns to z-scores.
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
    standard_scale :
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
    [row/col]_cluster : boolean
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
    [row/col]_linkage :
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
    [row/col]_colors :
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
    mask :
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
    """

    if sns is None:  # pragma: no cover
        raise Exception('seaborn not available. Please install seaborn.')

    if annot is not None:

        firstkey = list(annot.keys())[0]
        pal = sns.color_palette('hls', len(set(annot[firstkey])))
        lut = dict(zip(set(annot[firstkey]), pal))
        row_colors = [lut[k] for k in annot[firstkey]]

    _rs = {k: results[k]['value'] for k in results}
    data = pd.DataFrame.from_dict(_rs)
    if fform is not None:
        fform = fform
    else:
        fform = 'png'

    sns.clustermap(data,
                   method=method,
                   metric=metric,
                   z_score=z_score,
                   standard_scale=standard_scale,
                   row_cluster=row_cluster,
                   col_cluster=col_cluster,
                   row_linkage=row_linkage,
                   col_linkage=col_linkage,
                   col_colors=col_colors,
                   row_colors=row_colors,
                   mask=mask,
                   figsize=figsize,
                   **kwargs).savefig(os.path.join(output_dir,
                                                  name + '.' + fform),
                                     format=fform, dpi=700)


def export_tsne(output_dir, name, results, figsize=None,  # pylint: disable=too-many-locals
                cmap=None, colors=None, norm=None, alpha=None, fform=None,
                annot=None):
    """Create a plot of the 2D t-SNE embedding of the feature activities.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    fform : str or None
        Output file format. E.g. 'png', 'eps', etc. Default: 'png'.
    figsize : tuple(int, int) or None
        Used to specify the figure size for matplotlib.
    cmap : None or colormap
        Optional argument for matplotlib.pyplot.scatter. Only used if annot
        is abscent. Otherwise, marker colors are derived from the annotation.
    colors : None
        Optional argument for matplotlib.pyplot.scatter.
        Only used if annot
        is abscent. Otherwise, marker colors are derived from the annotation.
    norm :
        Optional argument for matplotlib.pyplot.scatter.
    alpha : None or float
        Opacity used for scatter plot markers.
    annot : None or dict.
        If annotation data is available, the color of the markers is automatically
        derived for the annotation.
    """

    if TSNE is None:  # pragma: no cover
        raise Exception('scikit-learn not available. '
                        'Please install scikit-learn to be able to use export_tsne.')
    if plt is None:  # pragma: no cover
        raise Exception('matplotlib not available. '
                        'Please install matplotlib to be able to use export_tsne.')

    _rs = {k: results[k]['value'] for k in results}
    data = pd.DataFrame.from_dict(_rs)

    tsne = TSNE()
    embedding = tsne.fit_transform(data.values)

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    if annot is not None:
        firstkey = list(annot.keys())[0]
        pal = sns.color_palette('hls', len(set(annot[firstkey])))
        lut = dict(zip(set(annot[firstkey]), pal))

        for label in lut:
            plt.scatter(x=embedding[np.asarray(annot[firstkey]) == label, 0],
                        y=embedding[np.asarray(annot[firstkey]) == label, 1],
                        c=lut[label],
                        label=label,
                        norm=norm, alpha=alpha)

        plt.legend()
    else:
        plt.scatter(x=embedding[:, 0],
                    y=embedding[:, 1],
                    c=colors, cmap=cmap,
                    norm=norm, alpha=alpha)

    plt.axis('off')
    if fform is not None:
        fform = fform
    else:
        fform = 'png'

    fig.savefig(os.path.join(output_dir, name + '.' + fform),
                format=fform, dpi=700)
