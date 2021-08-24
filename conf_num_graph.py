import csv
import argparse
import pandas as pd
import matplotlib.pyplot as plot

parser = argparse.ArgumentParser(description='Confidence Aware Learning')
parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to draw')
# parser.add_argument('--paths', default='./flood_graph/500_750/confidence_0.11_1000.csv', type=str, help='Path of csv')
parser.add_argument('--paths', default='./excel/cifar100/cifar100_idx_confidence_300.csv', type=str, help='Path of csv')
args = parser.parse_args()

def extract_data():
    print("Save Path : ", "/".join(args.paths.split("/")[:-1]))
    data = pd.read_csv('{}'.format(args.paths))
    df_bin = data.loc[:, [str(args.epochs)]]
    print(df_bin)
    df_bin *= 100
    avg = float(df_bin.mean())
    print(df_bin)
    print(df_bin.mean())
    print("**")



    bins = list(range(0, 101, 5))
    bins_label = [x for x in bins]
    print(bins_label)
    df_bin['bin'] = pd.cut(df_bin[str(args.epochs)], bins, right=False, labels=bins_label[:-1])
    print(bins)
    print(df_bin)
    print(df_bin.value_counts().sort_index(ascending=True))
    CountStatus = df_bin['bin'].value_counts().sort_index(ascending=True)
    CountStatus.plot.bar()
    CountStatus.plot.bar(grid = False, figsize = (10,8), fontsize = 15)
    plot.title("# per confidence in epoch {}".format(args.epochs), fontsize = 15)
    plot.text(5, 1, 'Avg Conf = {0:.3f}'.format(avg), fontsize=20, color='green')
    plot.ylim([0, 50000])

    plot.savefig("{}/{}.png".format("/".join(args.paths.split("/")[:-1]), args.epochs))
extract_data()