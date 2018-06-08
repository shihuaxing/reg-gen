import os
import numpy as np
from pysam import Samfile, Fastafile
from math import ceil, floor
from Bio import motifs
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyx
from scipy.stats.mvn import mvnun
from argparse import SUPPRESS
from scipy.signal import savgol_filter

from multiprocessing import Pool, cpu_count

# Internal
from rgt.Util import ErrorHandler, AuxiliaryFunctions, GenomeData, HmmData
from rgt.GenomicRegionSet import GenomicRegionSet
from rgt.HINT.biasTable import BiasTable

"""
Perform differential footprints analysis based on the prediction of transcription factor binding sites.

Authors: Eduardo G. Gusmao, Zhijian Li
"""

dic = {"A": 0, "C": 1, "G": 2, "T": 3}


def diff_analysis_args(parser):
    # Input Options
    parser.add_argument("--organism", type=str, metavar="STRING", default="hg19",
                        help="Organism considered on the analysis. Must have been setup in the RGTDATA folder. "
                             "Common choices are hg19, hg38. mm9, and mm10. DEFAULT: hg19")
    parser.add_argument("--window-size", type=int, metavar="INT", default=200,
                        help="The window size for differential analysis. DEFAULT: 200")

    parser.add_argument("--mpbs-files", type=str, metavar="FILE", default=None,
                        help="motif predicted binding sites files for all conditions, must be .bed file. DEFAULT: None")
    parser.add_argument("--reads-files", type=str, metavar="FILE", default=None,
                        help="The BAM files containing the DNase-seq or ATAC-seq reads for all conditions. "
                             "DEFAULT: None")
    parser.add_argument("--factors", type=float, metavar="FLOAT", default=None,
                        help="The normalization factors for conditions. DEFAULT: None")
    parser.add_argument("--names", type=str, metavar="STRING", default=None,
                        help="The name of conditions. DEFAULT: None")

    parser.add_argument("--fdr", type=float, metavar="FLOAT", default=0.05,
                        help="The false discovery rate. DEFAULT: 0.05")
    parser.add_argument("--bc", action="store_true", default=False,
                        help="If set, all analysis will be based on bias corrected signal. DEFAULT: False")
    parser.add_argument("--nc", type=int, metavar="INT", default=cpu_count(),
                        help="The number of cores. DEFAULT: 1")

    parser.add_argument("--forward-shift", type=int, metavar="INT", default=5, help=SUPPRESS)
    parser.add_argument("--reverse-shift", type=int, metavar="INT", default=-4, help=SUPPRESS)

    # Output Options
    parser.add_argument("--output-location", type=str, metavar="PATH", default=os.getcwd(),
                        help="Path where the output bias table files will be written. DEFAULT: current directory")
    parser.add_argument("--output-prefix", type=str, metavar="STRING", default="differential",
                        help="The prefix for results files. DEFAULT: differential")
    parser.add_argument("--standardize", action="store_true", default=False,
                        help="If set, the signal will be rescaled to (0, 1) for plotting.")
    parser.add_argument("--output-profiles", default=False, action='store_true',
                        help="If set, the footprint profiles will be writen into a text, in which each row is a "
                             "specific instance of the given motif. DEFAULT: False")


def get_raw_signal(arguments):
    (mpbs_name, mpbs, names, reads_files,
     organism, window_size, forward_shift, reverse_shift, bias_table) = arguments

    bam_dict = dict()
    signal_dict = dict()
    for i, name in enumerate(names):
        bam_dict[name] = Samfile(reads_files[i], "rb")
        signal_dict[name] = np.zeros(window_size)

    genome_data = GenomeData(organism)
    fasta = Fastafile(genome_data.get_genome())

    motif_len = None
    pwm = dict([("A", [0.0] * window_size), ("C", [0.0] * window_size),
                ("G", [0.0] * window_size), ("T", [0.0] * window_size),
                ("N", [0.0] * window_size)])

    mpbs_regions = mpbs.by_names([mpbs_name])
    num_motif = len(mpbs_regions)

    for region in mpbs_regions:
        if motif_len is None:
            motif_len = region.final - region.initial

        mid = (region.final + region.initial) / 2
        p1 = mid - window_size / 2
        p2 = mid + window_size / 2

        if p1 <= 0:
            continue

        # Fetch raw signal
        for name in names:
            for read in bam_dict[name].fetch(region.chrom, p1, p2):
                if not read.is_reverse:
                    cut_site = read.pos + forward_shift
                    if p1 <= cut_site < p2:
                        signal_dict[name][cut_site - p1] += 1.0
                else:
                    cut_site = read.aend + reverse_shift - 1
                    if p1 <= cut_site < p2:
                        signal_dict[name][cut_site - p1] += 1.0
        update_pwm(pwm, fasta, region, p1, p2)

    return signal_dict, motif_len, pwm, num_motif


def get_bc_signal(arguments):
    (mpbs_name, mpbs, names, reads_files,
     organism, window_size, forward_shift, reverse_shift, bias_table) = arguments

    bam_dict = dict()
    signal_dict = dict()
    for i, name in enumerate(names):
        bam_dict[name] = Samfile(reads_files[i], "rb")
        signal_dict[name] = np.zeros(window_size)

    genome_data = GenomeData(organism)
    fasta = Fastafile(genome_data.get_genome())

    motif_len = None
    pwm = dict([("A", [0.0] * window_size), ("C", [0.0] * window_size),
                ("G", [0.0] * window_size), ("T", [0.0] * window_size),
                ("N", [0.0] * window_size)])

    mpbs_regions = mpbs.by_names([mpbs_name])
    num_motif = len(mpbs_regions)

    # Fetch bias corrected signal
    for region in mpbs_regions:
        motif_len = region.final - region.initial

        mid = (region.final + region.initial) / 2
        p1 = mid - window_size / 2
        p2 = mid + window_size / 2

        if p1 <= 0:
            continue

        for name in names:
            signal = bias_correction(chrom=region.chrom, start=p1, end=p2, bam=bam_dict[name],
                                    bias_table=bias_table, genome_file_name=genome_data.get_genome(),
                                    forward_shift=forward_shift, reverse_shift=reverse_shift)
            if len(signal) != window_size:
                continue
            signal_dict[name] = np.add(signal_dict[name], np.array(signal))

        update_pwm(pwm, fasta, region, p1, p2)

    return signal_dict, motif_len, pwm, num_motif


def diff_analysis_run(args):
    # Initializing Error Handler
    err = ErrorHandler()

    output_location = os.path.join(args.output_location, "Lineplots")
    try:
        if not os.path.isdir(output_location):
            os.makedirs(output_location)
    except Exception:
        err.throw_error("MM_OUT_FOLDER_CREATION")

    # Parse the input files
    args.mpbs_files = args.mpbs_files.split(",")
    args.reads_files = args.reads_files.split(",")
    args.names = args.names.split(",")
    if args.factors:
        args.factors = args.factors.split(",")

    mpbs = GenomicRegionSet("Motif Predicted Binding Sites")
    for mpbs_file in args.mpbs_files:
        mpbs_tmp = GenomicRegionSet("MPBS")
        mpbs_tmp.read(mpbs_file)
        mpbs.combine(mpbs_tmp, output=False)

    mpbs.sort()
    factor_name_list = list(set(mpbs.get_names()))

    motif_len_dict = dict()
    motif_num_dict = dict()
    pwm_dict_by_tf = dict()

    pool = Pool(processes=args.nc)
    # differential analysis using bias corrected signal
    if args.bc:
        hmm_data = HmmData()
        table_F = hmm_data.get_default_bias_table_F_ATAC()
        table_R = hmm_data.get_default_bias_table_R_ATAC()
        bias_table = BiasTable().load_table(table_file_name_F=table_F, table_file_name_R=table_R)

        mpbs_list = list()
        for factor_name in factor_name_list:
            mpbs_list.append((factor_name, mpbs, args.names, args.reads_files, args.organism, args.window_size,
                args.forward_shift, args.reverse_shift, bias_table))
        res = pool.map(get_bc_signal, mpbs_list)

    # differential analysis using raw signal
    else:
        mpbs_list = list()
        for factor_name in factor_name_list:
            mpbs_list.append((factor_name, mpbs, args.names, args.reads_files,
                              args.organism, args.window_size, args.forward_shift, args.reverse_shift))
        res = pool.map(get_raw_signal, mpbs_list)

    signal_tf_dict = dict()
    for idx, factor_name in enumerate(factor_name_list):
        signal_tf_dict[factor_name] = res[idx][0]
        motif_len_dict[factor_name] = res[idx][1]
        pwm_dict_by_tf[factor_name] = res[idx][2]
        motif_num_dict[factor_name] = res[idx][3]

    # estimate the normalization factors
    if args.factors is None:
        args.factors = compute_factors(signal_tf_dict, factor_name_list, args.names)
        output_factor(args, args.factors)

    # normalize the signal
    for factor_name in factor_name_list:
        for idx, name in enumerate(args.names):
            signal_tf_dict[factor_name][name] = \
            ((signal_tf_dict[factor_name][name]) /  motif_num_dict[factor_name]) /  args.factors[idx]


    if args.output_profiles:
        output_profiles(signal_tf_dict, factor_name_list, args.names, output_location)

    plots_list = list()
    for factor_name in factor_name_list:
        signal_list = list()
        for name in args.names:
            signal_list.append(signal_tf_dict[factor_name][name])
        plots_list.append((factor_name, motif_num_dict[factor_name], signal_list, args.names,
                           pwm_dict_by_tf[factor_name], output_location, args.window_size, args.standardize))

    pool.map(line_plot, plots_list)

    ps_tc_results_by_tf = dict()
    for factor_name in factor_name_list:
        res = get_ps_tc_results(signal_tf_dict[factor_name], args.names, motif_num_dict[factor_name],
                                motif_len_dict[factor_name])
        ps_tc_results_by_tf[factor_name] = res

    if len(args.names) == 2:
        ps_tc_results_by_tf = get_stat_results(ps_tc_results_by_tf)
        scatter_plot(args, ps_tc_results_by_tf)
    output_stat_results(args, ps_tc_results_by_tf)


def bias_correction(chrom, start, end, bam, bias_table, genome_file_name, forward_shift, reverse_shift):
    # Parameters
    window = 50
    defaultKmerValue = 1.0

    # Initialization
    fastaFile = Fastafile(genome_file_name)
    fBiasDict = bias_table[0]
    rBiasDict = bias_table[1]
    k_nb = len(fBiasDict.keys()[0])
    p1 = start
    p2 = end
    p1_w = p1 - (window / 2)
    p2_w = p2 + (window / 2)
    p1_wk = p1_w - int(floor(k_nb / 2.))
    p2_wk = p2_w + int(ceil(k_nb / 2.))
    if (p1 <= 0 or p1_w <= 0 or p2_wk <= 0):
        # Return raw counts
        bc_signal = [0.0] * (p2 - p1)
        for read in bam.fetch(chrom, p1, p2):
            if not read.is_reverse:
                cut_site = read.pos + forward_shift
                if p1 <= cut_site < p2:
                    bc_signal[cut_site - p1] += 1.0
            else:
                cut_site = read.aend + reverse_shift - 1
                if p1 <= cut_site < p2:
                    bc_signal[cut_site - p1] += 1.0

        return bc_signal

    # Raw counts
    nf = [0.0] * (p2_w - p1_w)
    nr = [0.0] * (p2_w - p1_w)
    for read in bam.fetch(chrom, p1_w, p2_w):
        if not read.is_reverse:
            cut_site = read.pos + forward_shift
            if p1_w <= cut_site < p2_w:
                nf[cut_site - p1_w] += 1.0
        else:
            cut_site = read.aend + reverse_shift - 1
            if p1_w <= cut_site < p2_w:
                nr[cut_site - p1_w] += 1.0

    # Smoothed counts
    Nf = []
    Nr = []
    f_sum = sum(nf[:window])
    r_sum = sum(nr[:window])
    f_last = nf[0]
    r_last = nr[0]
    for i in range((window / 2), len(nf) - (window / 2)):
        Nf.append(f_sum)
        Nr.append(r_sum)
        f_sum -= f_last
        f_sum += nf[i + (window / 2)]
        f_last = nf[i - (window / 2) + 1]
        r_sum -= r_last
        r_sum += nr[i + (window / 2)]
        r_last = nr[i - (window / 2) + 1]

    # Fetching sequence
    currStr = str(fastaFile.fetch(chrom, p1_wk, p2_wk - 1)).upper()
    currRevComp = AuxiliaryFunctions.revcomp(str(fastaFile.fetch(chrom, p1_wk + 1, p2_wk)).upper())

    # Iterating on sequence to create signal
    af = []
    ar = []
    for i in range(int(ceil(k_nb / 2.)), len(currStr) - int(floor(k_nb / 2)) + 1):
        fseq = currStr[i - int(floor(k_nb / 2.)):i + int(ceil(k_nb / 2.))]
        rseq = currRevComp[len(currStr) - int(ceil(k_nb / 2.)) - i:len(currStr) + int(floor(k_nb / 2.)) - i]
        try:
            af.append(fBiasDict[fseq])
        except Exception:
            af.append(defaultKmerValue)
        try:
            ar.append(rBiasDict[rseq])
        except Exception:
            ar.append(defaultKmerValue)

    # Calculating bias and writing to wig file
    f_sum = sum(af[:window])
    r_sum = sum(ar[:window])
    f_last = af[0]
    r_last = ar[0]
    bc_signal = []
    for i in range((window / 2), len(af) - (window / 2)):
        nhatf = Nf[i - (window / 2)] * (af[i] / f_sum)
        nhatr = Nr[i - (window / 2)] * (ar[i] / r_sum)
        bc_signal.append(nhatf + nhatr)
        f_sum -= f_last
        f_sum += af[i + (window / 2)]
        f_last = af[i - (window / 2) + 1]
        r_sum -= r_last
        r_sum += ar[i + (window / 2)]
        r_last = ar[i - (window / 2) + 1]

    # Termination
    fastaFile.close()
    return bc_signal


def get_ps_tc_results(signal_dict, names, motif_len, window_size):
    signal_half_len = window_size / 2
    protect_score = list()
    tc = list()

    for name in names:
        nc = sum(signal_dict[name][signal_half_len - motif_len / 2:signal_half_len + motif_len / 2])
        nr = sum(signal_dict[name][signal_half_len + motif_len / 2:signal_half_len + motif_len / 2 + motif_len])
        nl = sum(signal_dict[name][signal_half_len - motif_len / 2 - motif_len:signal_half_len - motif_len / 2])
        ps = (nr - nc) / motif_len + (nl - nc) / motif_len
        tc = (sum(signal_dict[name]) - nc) / (len(signal_dict[name]) - motif_len)
        protect_score.append(ps)
        tc.append(tc)

    if len(names) == 2:
        diff_ps = protect_score[1] - protect_score[0]
        diff_tc = tc[1] - tc[0]
        return [protect_score, tc, diff_ps, diff_tc]
    else:
        return [protect_score, tc]


def update_pwm(pwm, fasta, region, p1, p2):
    # Update pwm
    aux_plus = 1
    dna_seq = str(fasta.fetch(region.chrom, p1, p2)).upper()
    if (region.final - region.initial) % 2 == 0:
        aux_plus = 0
    dna_seq_rev = AuxiliaryFunctions.revcomp(str(fasta.fetch(region.chrom,
                                                             p1 + aux_plus, p2 + aux_plus)).upper())
    if region.orientation == "+":
        for i in range(0, len(dna_seq)):
            pwm[dna_seq[i]][i] += 1
    elif region.orientation == "-":
        for i in range(0, len(dna_seq_rev)):
            pwm[dna_seq_rev[i]][i] += 1


def compute_factors(signal_tf_dict, factor_name_list, names):
    signal_dict = dict()

    for name in names:
        signal_dict[name] = np.zeros(len(factor_name_list))
        for idx, factor in enumerate(factor_name_list):
            signal_dict[name][idx] = sum(signal_tf_dict[factor][name])

    # Take log
    log_tc = dict()
    for name in names:
        log_tc[name] = np.log(signal_dict[name])

    # Average
    average_log_tc = np.zeros(len(factor_name_list))
    for name in names:
        average_log_tc = np.add(average_log_tc, log_tc[name])

    average_log_tc = average_log_tc / len(names)

    # Filter
    filter_log_tc = dict()
    for name in names:
        filter_log_tc[name] = log_tc[name][~np.isnan(log_tc[name])]
    average_log_tc = average_log_tc[~np.isnan(average_log_tc)]

    # Subtract
    sub_tc = dict()
    for name in names:
        sub_tc[name] = np.subtract(filter_log_tc[name], average_log_tc)

    median_tc = dict()
    for name in names:
        median_tc[name] = np.median(sub_tc[name])

    factors = list()
    for name in names:
        factors.append(np.exp(median_tc[name]))

    return factors


def line_plot(arguments):
    (factor_name, motif_num, signal_list, names, pwm_dict, output_location, window_size, standardize) = arguments

    factor_name = factor_name.replace("(", "_")
    factor_name = factor_name.replace(")", "")

    #if standardize:
    #    norm_signal = standard(norm_signal)

    # Output PWM and create logo
    pwm_fname = os.path.join(output_location, "{}.pwm".format(factor_name))
    pwm_file = open(pwm_fname, "w")
    for e in ["A", "C", "G", "T"]:
        pwm_file.write(" ".join([str(int(f)) for f in pwm_dict[e]]) + "\n")
    pwm_file.close()

    logo_fname = os.path.join(output_location, "{}.logo.eps".format(factor_name))
    pwm = motifs.read(open(pwm_fname), "pfm")
    pwm.weblogo(logo_fname, format="eps", stack_width="large", stacks_per_line=str(window_size),
                color_scheme="color_classic", unit_name="", show_errorbars=False, logo_title="",
                show_xaxis=False, xaxis_label="", show_yaxis=False, yaxis_label="",
                show_fineprint=False, show_ends=False)

    start = -(window_size / 2)
    end = (window_size / 2) - 1
    x = np.linspace(start, end, num=window_size)

    plt.close('all')
    fig, ax = plt.subplots()
    color_list = ['red', 'green', 'blue',
                  "#000000", "#000099", "#006600", "#990000", "#660099", "#CC00CC", "#222222", "#CC9900",
                  "#FF6600", "#0000CC", "#336633", "#CC0000", "#6600CC", "#FF00FF", "#555555", "#CCCC00",
                  "#FF9900", "#0000FF", "#33CC33", "#FF0000", "#663399", "#FF33FF", "#888888", "#FFCC00",
                  "#663300", "#009999", "#66CC66", "#FF3333", "#9933FF", "#FF66FF", "#AAAAAA", "#FFCC33",
                  "#993300", "#00FFFF", "#99FF33", "#FF6666", "#CC99FF", "#FF99FF", "#CCCCCC", "#FFFF00"]
    for idx, name in enumerate(names):
        ax.plot(x, signal_list[idx], color=color_list[idx], label=name)
    ax.text(0.15, 0.9, 'n = {}'.format(motif_num), verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontweight='bold')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 15))
    ax.tick_params(direction='out')
    ax.set_xticks([start, 0, end])
    ax.set_xticklabels([str(start), 0, str(end)])
    max_signal, min_signal = -np.inf, np.inf
    for idx, name in enumerate(names):
        if max_signal > max(signal_list[idx]):
            max_signal = max(signal_list[idx])
        if min_signal < min(signal_list[idx]):
            min_signal = min(signal_list[idx])
    ax.set_yticks([min_signal, max_signal])
    ax.set_yticklabels([str(round(min_signal, 2)), str(round(max_signal, 2))], rotation=90)

    ax.set_title(factor_name, fontweight='bold')
    ax.set_xlim(start, end)
    ax.set_ylim([min_signal, max_signal])
    ax.legend(loc="upper right", frameon=False)
    ax.spines['bottom'].set_position(('outward', 70))

    figure_name = os.path.join(output_location, "{}.line.eps".format(factor_name))
    fig.tight_layout()
    fig.savefig(figure_name, format="eps", dpi=300)

    # Creating canvas and printing eps / pdf with merged results
    output_fname = os.path.join(output_location, "{}.eps".format(factor_name))

    c = pyx.canvas.canvas()
    c.insert(pyx.epsfile.epsfile(0, 0, figure_name, scale=1.0))
    c.insert(pyx.epsfile.epsfile(0.45, 0.8, logo_fname, width=16.5, height=3))
    c.writeEPSfile(output_fname)
    os.system(" ".join(["epstopdf", output_fname]))

    os.remove(figure_name)
    os.remove(logo_fname)
    os.remove(output_fname)
    os.remove(pwm_fname)


def scatter_plot(args, stat_results_by_tf):
    tc_diff = list()
    ps_diff = list()
    mpbs_name_list = stat_results_by_tf.keys()
    P_values = list()
    for mpbs_name in mpbs_name_list:
        ps_diff.append(float(stat_results_by_tf[mpbs_name][2]))
        tc_diff.append(float(stat_results_by_tf[mpbs_name][-3]))
        P_values.append(np.log10(float(stat_results_by_tf[mpbs_name][-1])))

    fig, ax = plt.subplots(figsize=(12, 12))
    for i, mpbs_name in enumerate(mpbs_name_list):
        if stat_results_by_tf[mpbs_name][-1] < args.fdr:
            ax.scatter(tc_diff[i], ps_diff[i], c="red")
            ax.annotate(mpbs_name, (tc_diff[i], ps_diff[i]), alpha=0.6)
        else:
            ax.scatter(tc_diff[i], ps_diff[i], c="black", alpha=0.6)
    ax.margins(0.05)

    tc_diff_mean = np.mean(tc_diff)
    ps_diff_mean = np.mean(ps_diff)
    ax.axvline(x=tc_diff_mean, linewidth=2, linestyle='dashed')
    ax.axhline(y=ps_diff_mean, linewidth=2, linestyle='dashed')

    ax.set_xlabel("{} $\longrightarrow$ {} \n $\Delta$ Open Chromatin Score".format(args.condition1, args.condition2),
                  fontweight='bold', fontsize=20)
    ax.set_ylabel("$\Delta$ Protection Score \n {} $\longrightarrow$ {}".format(args.condition1, args.condition2),
                  fontweight='bold', rotation=90, fontsize=20)

    figure_name = os.path.join(args.output_location, "{}_{}_statistics.pdf".format(args.condition1, args.condition2))
    fig.savefig(figure_name, format="pdf", dpi=300)


def scatter_plot3(args, stat_results_by_tf):
    tc_diff = list()
    ps_diff = list()
    mpbs_name_list = stat_results_by_tf.keys()
    P_values = list()
    for mpbs_name in mpbs_name_list:
        ps_diff.append(float(stat_results_by_tf[mpbs_name][2]))
        tc_diff.append(float(stat_results_by_tf[mpbs_name][-3]))
        P_values.append(np.log10(float(stat_results_by_tf[mpbs_name][-1])))

    fig, ax = plt.subplots(figsize=(12, 12))
    for i, mpbs_name in enumerate(mpbs_name_list):
        if stat_results_by_tf[mpbs_name][-1] < args.fdr:
            ax.scatter(tc_diff[i], ps_diff[i], c="red")
            ax.annotate(mpbs_name, (tc_diff[i], ps_diff[i]), alpha=0.6)
        else:
            ax.scatter(tc_diff[i], ps_diff[i], c="black", alpha=0.6)
    ax.margins(0.05)

    tc_diff_mean = np.mean(tc_diff)
    ps_diff_mean = np.mean(ps_diff)
    ax.axvline(x=tc_diff_mean, linewidth=2, linestyle='dashed')
    ax.axhline(y=ps_diff_mean, linewidth=2, linestyle='dashed')

    ax.set_xlabel("{} $\longrightarrow$ {} \n $\Delta$ Open Chromatin Score".format(args.condition1, args.condition2),
                  fontweight='bold', fontsize=20)
    ax.set_ylabel("$\Delta$ Protection Score \n {} $\longrightarrow$ {}".format(args.condition1, args.condition2),
                  fontweight='bold', rotation=90, fontsize=20)

    figure_name = os.path.join(args.output_location, "{}_{}_statistics.pdf".format(args.condition1, args.condition2))
    fig.savefig(figure_name, format="pdf", dpi=300)


def output_stat_results(args, ps_tc_results_by_tf):
    output_fname = os.path.join(args.output_location, "{}_statistics.txt".format(args.output_prefix))
    header = ["Motif"]
    for name in args.names:
        header.append("Protection_Score_{}".format(name))
        header.append("Tag_Count_{}".format(name))

    if len(args.names) > 2:
        with open(output_fname, "w") as f:
            f.write("\t".join(header) + "\n")
            for factor_name in ps_tc_results_by_tf.keys():
                ps_list = ps_tc_results_by_tf[factor_name][0]
                tc_list = ps_tc_results_by_tf[factor_name][1]
                f.write(factor_name + "\t" + "\t".join(map(str, ps_list)) + "\t".join(map(str, tc_list)) + "\n")
    else:
        header.append("Protection_Diff_{}_{}".format(args.names[0], args.names[1]))
        header.append("TC_Diff_{}_{}".format(args.names[0], args.names[1]))
        with open(output_fname, "w") as f:
            f.write("\t".join(header) + "\n")
            for factor_name in ps_tc_results_by_tf.keys():
                ps_list = ps_tc_results_by_tf[factor_name][0]
                tc_list = ps_tc_results_by_tf[factor_name][1]
                f.write(factor_name + "\t" + "\t".join(map(str, ps_list)) + "\t".join(map(str, tc_list)) + "\t" +
                        ps_tc_results_by_tf[factor_name][2] + "\t" + ps_tc_results_by_tf[factor_name][3] + "\n")


def output_factor(args, factors):
    output_file = os.path.join(args.output_location, "{}_factor.txt".format(args.output_prefix))
    with open(output_file, "w") as f:
        for idx, name in enumerate(args.names):
            f.write("Factor of {}: ".format(name) + str(factors[idx]) + "\n")


def output_mu(args, median_diff_prot, median_diff_tc):
    output_file = os.path.join(args.output_location, "{}_{}_mu.txt".format(args.condition1, args.condition2))
    f = open(output_file, "w")
    f.write("median_diff_prot: " + str(median_diff_prot) + "\n")
    f.write("median_diff_tc: " + str(median_diff_tc) + "\n")
    f.close()


def get_stat_results(ps_tc_results_by_tf):
    ps_diff = list()
    tc_diff = list()
    mpbs_name_list = ps_tc_results_by_tf.keys()
    for mpbs_name in mpbs_name_list:
        ps_diff.append(ps_tc_results_by_tf[mpbs_name][2])
        tc_diff.append(ps_tc_results_by_tf[mpbs_name][-1])

    ps_tc_diff = np.array([ps_diff, tc_diff]).T
    mu = np.mean(ps_tc_diff, axis=0)
    cov_ps_tc_diff = np.cov(ps_tc_diff.T)

    low = np.zeros(2)
    upp = np.zeros(2)
    p_values = list()
    for idx, mpbs_name in enumerate(mpbs_name_list):
        if ps_diff[idx] >= mu[0]:
            low[0] = ps_diff[idx]
            upp[0] = float('inf')
        else:
            low[0] = -float('inf')
            upp[0] = ps_diff[idx]

        if tc_diff[idx] >= mu[1]:
            low[1] = tc_diff[idx]
            upp[1] = float('inf')
        else:
            low[1] = -float('inf')
            upp[1] = tc_diff[idx]

        p_value, i = mvnun(low, upp, mu, cov_ps_tc_diff)
        ps_tc_results_by_tf[mpbs_name].append(p_value)
        p_values.append(p_value)

    adjusted_p_values = adjust_p_values(p_values)
    for idx, mpbs_name in enumerate(mpbs_name_list):
        ps_tc_results_by_tf[mpbs_name].append(adjusted_p_values[idx])

    return ps_tc_results_by_tf


def standard(vector1, vector2):
    max_ = max(max(vector1), max(vector2))
    min_ = min(min(vector1), min(vector2))
    if max_ > min_:
        return [(e - min_) / (max_ - min_) for e in vector1], [(e - min_) / (max_ - min_) for e in vector2]
    else:
        return vector1, vector2


def adjust_p_values(p_values):
    p = np.asfarray(p_values)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


def output_profiles(signal_tf_dict, factor_name_list, names, output_location):
    for name in names:
        for factor_name in factor_name_list:
            output_fname = os.path.join(output_location, "{}_{}.txt".format(factor_name, name))
            with open(output_fname, "w") as f:
                f.write("\t".join(map(str, signal_tf_dict[factor_name][name])) + "\n")
