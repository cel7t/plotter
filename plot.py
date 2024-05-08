#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textalloc as ta
from pyfzf.pyfzf import FzfPrompt
# To suppress the palette warning
import warnings
warnings.filterwarnings("ignore")

debug_flag = False

def parse_rqtl_file(file_path, trait, marker_file):
    df0 = pd.read_csv(file_path, index_col=0, sep="	")
    mf = pd.read_csv(marker_file, sep="\t")
    if debug_flag:
        print(mf)
    if trait is None:
        traits = df0.index.values.tolist()
        fzf = FzfPrompt()
        trait = fzf.prompt(traits)
    df = df0.loc[trait]
    def get_chr(marker):
        return marker.split("_")[1]
    def get_pos(marker):
        return mf.loc[mf["marker"] == marker, "start"].values[0]
    df = df.T.reset_index()
    df.columns = ["marker", "LOD"]
    df['LOD'] = df['LOD'].apply(lambda x: abs(x))
    df["chr"] = df["marker"].apply(lambda x: get_chr(x))
    df["pos"] = df["marker"].apply(lambda x: get_pos(x))
    if debug_flag:
        print(df, trait)
    return (df, trait)

def draw_qtl_plot(df, draw_peak, threshold_value, hovering_enabled, marker_data, centromere_data, two_color, trait, snp_density, bin_size):
    df.chr = df.chr.astype('category')
    category_order = df.chr.unique()
    df.chr = df.chr.cat.set_categories(category_order, ordered=True)
    shiftpos = 0
    cpos = []
    for _, group_df in df.groupby('chr', observed=False):
        cpos.append(group_df['pos'] + shiftpos)
        shiftpos += group_df['pos'].max()
    df['cpos'] = pd.concat(cpos)
    sns.set_theme()
    sns.set_style(rc = {'axes.facecolor': "#eeeeee", 'grid.color': "#f0f0f0"})
    qtl_palette = ['#0173b2',
                         '#de8f05'] if two_color else ['#0173b2', '#de8f05',
                                                       '#029e73', '#d55e00',
                                                       '#cc78bc', '#ca9161',
                                                       '#56b4e9', '#949494']
    qtl_plot = sns.relplot(
        data=df,
        alpha=0.7,
        x='cpos',
        y='LOD',
        hue='chr',
        palette=qtl_palette,
        linewidth=1,
        legend=None,
        kind='line'
       )
    qtl_plot.set(title=trait[0])
    for line in qtl_plot.ax.lines:
        x, y = line.get_xydata().T
        qtl_plot.ax.fill_between(x, -0.05, y, color=line.get_color(), alpha=0.3)
    qtl_plot.ax.set_ylim(df['LOD'].min() - 0.05, df['LOD'].max() + 0.05)
    qtl_plot.ax.set_ylabel('LOD', rotation=0, labelpad=24)
    cpos_spacing = (df.groupby('chr', observed=False)['cpos'].max()).iloc[0]
    cpos_spacing = cpos_spacing - (df.groupby('chr', observed=False)['cpos'].min()).iloc[0]
    cpos_spacing = cpos_spacing/20
    qtl_plot.ax.set_xlim(df['cpos'].min() - cpos_spacing, df['cpos'].max() + cpos_spacing)
    if len(df["chr"].unique()) > 1:
        qtl_plot.ax.set_xlabel('Chromosome')
        qtl_plot.ax.set_xticks(df.groupby('chr', observed=False)['cpos'].median())
        qtl_plot.ax.xaxis.grid(False)
        qtl_plot.ax.set_xticklabels(df['chr'].unique())
        qtl_plot.ax.xaxis.grid(False)
        qtl_plot.ax.set_xticklabels(df['chr'].unique())
    else:
        qtl_plot.ax.set_xlabel('position')
        xtick_step = len(df['pos']) // 10
        qtl_plot.ax.set_xticks(df['pos'][::xtick_step])
    prev_tick = 0.0
    span_color = 'lightgrey'
    for idx, tick in enumerate(df.groupby('chr', observed=False)['cpos'].min()):
        if debug_flag:
            print(idx, tick)
        if idx != 'N/A':
            qtl_plot.ax.axvspan(prev_tick, tick, facecolor=span_color, zorder=0, alpha=0.5)
        prev_tick = tick
        span_color = '#ccccee' if span_color == 'lightgrey' else 'lightgrey'
    last_tick = (df.groupby('chr', observed=False)['cpos'].max()).iloc[-1]
    qtl_plot.ax.axvspan(prev_tick, last_tick, facecolor=span_color, zorder=0, alpha=0.5)
    plt.subplots_adjust(bottom=0.1, left=0.07, top=0.95, right=0.97)
    if draw_peak:
        maxlp = df.loc[df['LOD'].idxmax()]
        qtl_plot.ax.axvline(x=maxlp['cpos'],
                                  color=sns.color_palette('deep')[3],
                                  linestyle='dashed',
                                  linewidth=1)
    if threshold_value:
        qtl_plot.ax.axhline(y=threshold_value,
                                  color=sns.color_palette('deep')[3],
                                  linestyle='dashed',
                                  linewidth=1)
    if centromere_data is not None:
        existing_chromosomes = df["chr"].unique()
        for chr in centromere_data["chr"]:
            if debug_flag:
                print("Centromere for chromosome", chr)
            if chr[3:] in existing_chromosomes:
                arm_length = centromere_data.loc[centromere_data["chr"] == chr, "arm_len"]
                arm_length = arm_length.iloc[-1]
                chr_pad = df[df["chr"] == chr[3:]]["cpos"].min()
                arm_length += chr_pad
                qtl_plot.ax.axvline(x=arm_length,
                                          color=sns.color_palette('deep')[0],
                                          linestyle='dashed',
                                          linewidth=1)
    if snp_density is not False and len(df["chr"].unique()) < 2:
        df_start = df["pos"].min()
        df_end = df["pos"].max()
        df_range = df_end - df_start
        if debug_flag:
            print(df_start, df_end, df_range)
        df_blk_size = bin_size
        num_of_blks = df_range//df_blk_size
        df_blks = np.zeros(num_of_blks)
        for i in range(num_of_blks):
            start_idx = df_start + i * df_blk_size
            end_idx = min(df_start + (i+1) * df_blk_size, df_end)
            vals_idx = df[df["pos"].between(start_idx, end_idx)]
            df_blks[i] = int(vals_idx.shape[0])
        if debug_flag:
            print(df_blks)
        plt.figure()
        plt.plot(np.arange(num_of_blks), df_blks)
        plt.xlabel('Blocks')
        plt.ylabel('Block Values')
        plt.grid(True)
    text_list = []
    x_list = []
    y_list = []
    # As this is a line plot, we need to skip one line each for each chromosome!
    skip_lines = (not not threshold_value) + (not not draw_peak) + len(df.chr.unique()) + (not not snp_density)
    if debug_flag:
        print("Skipping", skip_lines, "lines.")
    def clear_points_and_lines():
        for line_idx, line in enumerate(plt.gca().lines):
            if line_idx < skip_lines:
                if debug_flag:
                    print("Skipped a line")
                continue
            line.remove()
        for text in qtl_plot.ax.texts:
            if text != hover_annot:
                text.remove()
    def markers_from_file(marker_name_list):
        for marker_name in marker_name_list:
            if debug_flag:
                print(marker_name)
            marker_idx = df.loc[df['marker'] == marker_name]
            text_attribute = marker_idx['marker'].iloc[0]
            x_attribute = marker_idx['cpos'].iloc[0]
            y_attribute = marker_idx['LOD'].iloc[0]
            text_list.append(text_attribute)
            x_list.append(x_attribute)
            y_list.append(y_attribute)
        ta.allocate_text(fig=qtl_plot.figure,
                                     ax=qtl_plot.ax,
                                     x=x_list,
                                     y=y_list,
                                     text_list=text_list,
                                     linecolor=sns.color_palette('deep')[3],
                                     textsize=12)
        plt.draw()
    if marker_data is not None:
        markers_from_file(marker_data)
    def on_click(event):
        if event.button == 3 and hovering_enabled:
            create_or_destroy_hover_annot()
            qtl_plot.fig.canvas.draw_idle()
            return
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            if debug_flag:
                print(x, y)
            closest_point_index = (((df['cpos'] - x)/df['cpos'])**2 + ((df['LOD'] - y)/df['LOD'])**2).idxmin()
            marker_attribute = df.loc[closest_point_index, 'marker']
            x_attribute = df.loc[closest_point_index, 'cpos']
            y_attribute = df.loc[closest_point_index, 'LOD']
            if debug_flag:
                print(x_attribute, y_attribute)
                print(f"Clicked on point with marker attribute: {marker_attribute}")
            if marker_attribute not in text_list:
                if debug_flag:
                    print("New attribute!")
                text_list.append(marker_attribute)
                x_list.append(x_attribute)
                y_list.append(y_attribute)
            else:
                if debug_flag:
                    print("Existing attribute: deleting!")
                idx = text_list.index(marker_attribute)
                deleted_flag = False
                for text_obj in qtl_plot.ax.texts:
                    if text_obj.get_text() == marker_attribute:
                        text_obj.remove()
                        if debug_flag:
                            print("Found the text object!")
                        deleted_flag = True
                        break
                if not deleted_flag:
                    print("ERROR: Deleted Flag not satisfied for point!")
                else:
                    deleted_flag = False
                text_list.pop(idx)
                x_list.pop(idx)
                y_list.pop(idx)
            clear_points_and_lines()
            ta.allocate_text(fig=qtl_plot.figure,
                                         ax=qtl_plot.ax,
                                         x=x_list,
                                         y=y_list,
                                         text_list=text_list,
                                         linecolor=sns.color_palette('deep')[3],
                                         textsize=12)
            plt.draw()
    hover_annot = qtl_plot.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                             textcoords="offset points",
                                             bbox=dict(boxstyle="round",
                                                       fc=(0.94, 0.95, 0.9)),
                                             arrowprops=dict(arrowstyle="->",
                                                             color="b"))
    if not hovering_enabled:
        hover_annot.set_visible(False)
    def create_or_destroy_hover_annot():
        nonlocal hover_annot
        if hover_annot.get_visible():
            hover_annot.set_visible(False)
        else:
            hover_annot.set_visible(True)
    def update_hover_annot(event):
        x, y = event.xdata, event.ydata
        if debug_flag:
            print(x, y)
        closest_point_index = (((df['cpos'] - x)/df['cpos'])**2 + ((df['LOD'] - y)/df['LOD'])**2).idxmin()
        marker_attribute = df.loc[closest_point_index, 'marker']
        x_attribute = df.loc[closest_point_index, 'cpos']
        y_attribute = df.loc[closest_point_index, 'LOD']
        txt = "Pos: "+str(x_attribute)+", LOD: "+str(y_attribute)+": "+marker_attribute
        hover_annot.set_text(txt)
        hover_annot.xy = (x_attribute, y_attribute)
        hover_annot.get_bbox_patch().set_alpha(0.4)

    def on_hover(event):
        if not hover_annot.get_visible():
            return
        update_hover_annot(event)
        qtl_plot.fig.canvas.draw_idle()

    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    if hovering_enabled:
        plt.gcf().canvas.mpl_connect('motion_notify_event', on_hover)
    plt.show()

def parse_csv_file(file_path):
    col_names = ["chr", "pos", "marker", "af", "beta", "se", "l_mle", "-logP"]
    df = pd.read_csv(file_path, sep=',', header=0, names=col_names)
    return df

def parse_lmdb_file(file_path):
    print("Not yet implemented")
    return 0

def draw_manhattan_plot(df, draw_peak, threshold_value, hovering_enabled, marker_data, centromere_data, two_color, snp_density, bin_size):
    df.chr = df.chr.astype('category')
    category_order = df.chr.unique()
    df.chr = df.chr.cat.set_categories(category_order, ordered=True)
    shiftpos = 0
    cpos = []
    for chromosome, group_df in df.groupby('chr', observed=False):
        cpos.append(group_df['pos'] + shiftpos)
        shiftpos += group_df['pos'].max()
    df['cpos'] = pd.concat(cpos)
    sns.set_theme()
    sns.set_style(rc = {'axes.facecolor': "#eeeeee", 'grid.color': "#f0f0f0"})
    manhattan_palette = ['#0173b2',
                         '#de8f05'] if two_color else ['#0173b2', '#de8f05',
                                                       '#029e73', '#d55e00',
                                                       '#cc78bc', '#ca9161',
                                                       '#56b4e9', '#949494']
    manhattan_plot = sns.relplot(
        data=df,
        alpha=0.7,
        x='cpos',
        y='-logP',
        hue='chr',
        palette=manhattan_palette,
        linewidth=0,
        legend=None
       )
    manhattan_plot.ax.set_ylim(-0.05, None)
    manhattan_plot.ax.set_ylabel('-log P', rotation=0, labelpad=24)
    cpos_spacing = (df.groupby('chr', observed=False)['cpos'].max()).iloc[0]
    cpos_spacing = cpos_spacing - (df.groupby('chr', observed=False)['cpos'].min()).iloc[0]
    cpos_spacing = cpos_spacing/20
    manhattan_plot.ax.set_xlim(df['cpos'].min() - cpos_spacing, df['cpos'].max() + cpos_spacing)
    if len(df["chr"].unique()) > 1:
        manhattan_plot.ax.set_xlabel('Chromosome')
        manhattan_plot.ax.set_xticks(df.groupby('chr', observed=False)['cpos'].median())
        manhattan_plot.ax.xaxis.grid(False)
        manhattan_plot.ax.set_xticklabels(df['chr'].unique())
    else:
        manhattan_plot.ax.set_xlabel('position')
        xtick_step = len(df['pos']) // 10
        manhattan_plot.ax.set_xticks(df['pos'][::xtick_step])
    prev_tick = 0.0
    span_color = 'lightgrey'
    for idx, tick in enumerate(df.groupby('chr', observed=False)['cpos'].min()):
        if debug_flag:
            print("Enumerating:", idx, tick)
        if idx != 'N/A':
            manhattan_plot.ax.axvspan(prev_tick, tick, facecolor=span_color, zorder=0, alpha=0.5)
        prev_tick = tick
        span_color = '#ccccee' if span_color == 'lightgrey' else 'lightgrey'
    last_tick = (df.groupby('chr', observed=False)['cpos'].max()).iloc[-1]
    manhattan_plot.ax.axvspan(prev_tick, last_tick, facecolor=span_color, zorder=0, alpha=0.5) # alpha=0.3
    plt.subplots_adjust(bottom=0.1, left=0.1, top=0.95, right=0.9)
    if draw_peak:
        maxlp = df.loc[df['-logP'].idxmax()]
        manhattan_plot.ax.axvline(x=maxlp['cpos'],
                                  color=sns.color_palette('deep')[3],
                                  linestyle='dashed',
                                  linewidth=1)
    if threshold_value:
        manhattan_plot.ax.axhline(y=threshold_value,
                                  color=sns.color_palette('deep')[3],
                                  linestyle='dashed',
                                  linewidth=1)
    centromere_lines = 0
    if centromere_data is not None:
        existing_chromosomes = df["chr"].unique()
        for chr in centromere_data["chr"]:
            if debug_flag:
                print("Centromere for chromosome", chr)
            if chr[3:] in existing_chromosomes:
                centromere_lines += 1
                arm_length = centromere_data.loc[centromere_data["chr"] == chr, "arm_len"]
                arm_length = arm_length.iloc[-1]
                chr_pad = df[df["chr"] == chr[3:]]["cpos"].min()
                arm_length += chr_pad
                manhattan_plot.ax.axvline(x=arm_length,
                                          color=sns.color_palette('deep')[0],
                                          linestyle='dashed',
                                          linewidth=1)
    if snp_density is not False and len(df["chr"].unique()) < 2:
        df_start = df["pos"].min()
        df_end = df["pos"].max()
        df_range = df_end - df_start
        if debug_flag:
            print("Start:", df_start, "End:", df_end, "Range:", df_range)
        df_blk_size = bin_size
        num_of_blks = df_range//df_blk_size
        df2 = pd.DataFrame({'snp': [0]*num_of_blks, 'blkpos': [0]*num_of_blks})
        for i in range(num_of_blks):
            start_idx = df_start + i * df_blk_size
            end_idx = min(df_start + (i+1) * df_blk_size, df_end)
            snp_val = (df[df["pos"].between(start_idx, end_idx)]).shape[0]
            df2["blkpos"][i] = start_idx
            df2["snp"][i] = snp_val
        max_snp = float(df2['snp'].max())
        df2['snp'] = (df2['snp'] / max_snp) * 0.2 - 0.05
        sns.lineplot(x='blkpos', y='snp', data=df2, ax=manhattan_plot.ax, color='#de8f05')
    text_list = []
    x_list = []
    y_list = []
    skip_lines = (not not threshold_value) + (not not draw_peak) + centromere_lines + (not not snp_density)
    def clear_points_and_lines():
        for line_idx, line in enumerate(plt.gca().lines):
            if line_idx < skip_lines:
                if debug_flag:
                    print("Skipped a line")
                continue
            line.remove()
        for text in manhattan_plot.ax.texts:
            if text != hover_annot:
                text.remove()
    def markers_from_file(marker_name_list):
        for marker_name in marker_name_list:
            if debug_flag:
                print(marker_name)
            marker_idx = df.loc[df['marker'] == marker_name]
            text_attribute = marker_idx['marker'].iloc[0]
            x_attribute = marker_idx['cpos'].iloc[0]
            y_attribute = marker_idx['-logP'].iloc[0]
            text_list.append(text_attribute)
            x_list.append(x_attribute)
            y_list.append(y_attribute)
        ta.allocate_text(fig=manhattan_plot.figure,
                                     ax=manhattan_plot.ax,
                                     x=x_list,
                                     y=y_list,
                                     text_list=text_list,
                                     linecolor=sns.color_palette('deep')[3],
                                     textsize=12)
        plt.draw()
    if marker_data is not None:
        markers_from_file(marker_data)
    def on_click(event):
        if event.button == 3 and hovering_enabled:
            create_or_destroy_hover_annot()
            manhattan_plot.fig.canvas.draw_idle()
            return
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            if debug_flag:
                print(x, y)
            closest_point_index = (((df['cpos'] - x)/df['cpos'])**2 + ((df['-logP'] - y)/df['-logP'])**2).idxmin()
            marker_attribute = df.loc[closest_point_index, 'marker']
            x_attribute = df.loc[closest_point_index, 'cpos']
            y_attribute = df.loc[closest_point_index, '-logP']
            if debug_flag:
                print(x_attribute, y_attribute)
                print(f"Clicked on point with marker attribute: {marker_attribute}")
            if marker_attribute not in text_list:
                if debug_flag:
                    print("New attribute!")
                text_list.append(marker_attribute)
                x_list.append(x_attribute)
                y_list.append(y_attribute)
            else:
                if debug_flag:
                    print("Existing attribute: deleting!")
                idx = text_list.index(marker_attribute)
                deleted_flag = False
                for text_obj in manhattan_plot.ax.texts:
                    if text_obj.get_text() == marker_attribute:
                        text_obj.remove()
                        if debug_flag:
                            print("Found the text object!")
                        deleted_flag = True
                        break
                if not deleted_flag:
                    print("ERROR: Deleted Flag not satisfied for point!")
                else:
                    deleted_flag = False
                text_list.pop(idx)
                x_list.pop(idx)
                y_list.pop(idx)
            clear_points_and_lines()
            ta.allocate_text(fig=manhattan_plot.figure,
                                         ax=manhattan_plot.ax,
                                         x=x_list,
                                         y=y_list,
                                         text_list=text_list,
                                         linecolor=sns.color_palette('deep')[3],
                                         textsize=12)
            plt.draw()
    hover_annot = manhattan_plot.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                             textcoords="offset points",
                                             bbox=dict(boxstyle="round",
                                                       fc=(0.94, 0.95, 0.9)),
                                             arrowprops=dict(arrowstyle="->",
                                                             color="b"))
    if not hovering_enabled:
        hover_annot.set_visible(False)
    def create_or_destroy_hover_annot():
        nonlocal hover_annot
        if hover_annot.get_visible():
            hover_annot.set_visible(False)
        else:
            hover_annot.set_visible(True)
    def update_hover_annot(event):
        x, y = event.xdata, event.ydata
        if debug_flag:
            print(x, y)
        closest_point_index = (((df['cpos'] - x)/df['cpos'])**2 + ((df['-logP'] - y)/df['-logP'])**2).idxmin()
        marker_attribute = df.loc[closest_point_index, 'marker']
        x_attribute = df.loc[closest_point_index, 'cpos']
        y_attribute = df.loc[closest_point_index, '-logP']
        txt = "Pos: "+str(x_attribute)+", -logP: "+str(y_attribute)+": "+marker_attribute
        hover_annot.set_text(txt)
        hover_annot.xy = (x_attribute, y_attribute)
        hover_annot.get_bbox_patch().set_alpha(0.4)

    def on_hover(event):
        if not hover_annot.get_visible():
            return
        update_hover_annot(event)
        manhattan_plot.fig.canvas.draw_idle()

    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    if hovering_enabled:
        plt.gcf().canvas.mpl_connect('motion_notify_event', on_hover)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files based on their extensions.')
    parser.add_argument('file', type=str, help='Path to the file to process')
    parser.add_argument('--two-color',
                        help='Use only two colors for the plot',
                        action='store_true',
                        default=False)
    parser.add_argument('--peak',
                        help='Draw a vertical line through the peak value',
                        action='store_true')
    parser.add_argument('--threshold',
                        type=float,
                        help='Draw a threshold line at a given -logP value')
    parser.add_argument('--hover',
                        help='Show details of the point that the cursor is hovering on',
                        action='store_true')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debugging', default=False)
    parser.add_argument('--line', action='store_true', help='QTL plot', default=False)
    parser.add_argument('--snp-density', action='store_true', help='Plot SNP density', default=False)
    parser.add_argument('--markers', default=None, nargs='?', type=str, help='Path to the markers to be highlighted')
    parser.add_argument('--marker-file', default=None, nargs='?', type=str, help='Path to the marker file')
    parser.add_argument('--trait', default=None, nargs='?', type=str, help='Trait name for AraQTL file')
    parser.add_argument('--chromosome', default=None, nargs='?', type=str, help='Selected chromosome')
    parser.add_argument('--bin-size', default=100000, nargs='?', type=int, help='Bin size for SNP density')
    parser.add_argument('--centromeres', default=None, nargs='?', type=str, help='Path to the centromere file')
    args = parser.parse_args()
    if args.line is True:
        data, args.trait = parse_rqtl_file(args.file, args.trait, args.marker_file)
    else:
        _, file_extension = os.path.splitext(args.file)
        if file_extension.lower() == '.csv':
            data = parse_csv_file(args.file)
        elif file_extension.lower() == '.mdb':
            data = parse_lmdb_file(args.file)
        else:
            print(f"Unsupported file extension: {file_extension}. Please provide a CSV or MDB file.")
            exit(1)
    debug_flag = args.debug
    if args.chromosome:
        data = data[data["chr"] == args.chromosome]
        if debug_flag:
            print("Succeeded in picking chromosome!")
    if args.markers:
        with open(args.markers, 'r') as marker_file:
            marker_data = [line.strip() for line in marker_file]
    else:
        marker_data = None
    if args.centromeres:
        centromere_col_names = ["chr", "arm_type", "arm_len"]
        centromere_data = pd.read_csv(args.centromeres, sep=' ', header=0, names=centromere_col_names)
    else:
        centromere_data = None
    if args.line:
        draw_qtl_plot(data, args.peak, args.threshold, args.hover, marker_data, centromere_data, args.two_color, args.trait, args.snp_density, args.bin_size)
    else:
        draw_manhattan_plot(data, args.peak, args.threshold, args.hover, marker_data, centromere_data, args.two_color, args.snp_density, args.bin_size)
