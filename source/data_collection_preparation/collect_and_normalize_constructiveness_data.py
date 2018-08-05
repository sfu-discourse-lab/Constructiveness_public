import argparse
from constructiveness_data_collector import *


def get_arguments():
    parser = argparse.ArgumentParser(description='Create training and test datafiles for constructiveness')

    parser.add_argument('--nyt_picks_csv', '-np', type=str, dest='nyt_picks_csv', action='store',
                        default='/Users/vkolhatk/Data/Constructiveness/data/NYT_comments_csv/NYT_comments_all_June_6_editor_picks.csv',
                        help="The NYT picks csv file collected using the NYT API")

    parser.add_argument('--commentIQ_nyt_picks_csv', '-cnp', type=str, dest='commentIQ_nyt_picks_csv', action='store',
                        default='/Users/vkolhatk/Data/Constructiveness/data/NYT_comments_csv/editorsselections-10-2014.csv',
                        help="The commentIQ NYT picks csv file")

    parser.add_argument('--YNC_expert_CSV', '-ye', type=str, dest='ync_expert_path', action='store',
                        default='/Users/vkolhatk/Data/ydata-ynacc-v1_0/ydata-ynacc-v1_0_expert_annotations.tsv',
                        help="The path of the CSV containing Yahoo News Annotated Comments Corpus (YNACC) expert annotations.")

    parser.add_argument('--YNC_mturk_CSV', '-yt', type=str, dest='ync_mturk_path', action='store',
                        default='/Users/vkolhatk/Data/ydata-ynacc-v1_0/ydata-ynacc-v1_0_turk_annotations.tsv',
                        help="The path of the CSV containing Yahoo News Annotated Comments Corpus (YNACC) mturk annotations.")

    parser.add_argument('--YNC_IAC_CSV', '-yi', type=str, dest='ync_IAC_path', action='store',
                        default='/Users/vkolhatk/Data/ydata-ynacc-v1_0/ydata-ynacc-v1_0_turk_annotations.tsv',
                        help="The path of the CSV containing Yahoo News Annotated Comments Corpus (YNACC) IAC annotations.")

    parser.add_argument('--SOCC_annotated_csv', '-SOCC_subset', type=str, dest='SOCC_annotated_csv', action='store',
                        default='/Users/vkolhatk/Data/SOCC/annotated/constructiveness/SFU_constructiveness_toxicity_corpus.csv',
                        help="SOCC constructiveness and toxicity annotations CSV")

    parser.add_argument('--train_csv', '-tr', type=str, dest='train_csv', action='store',
                        default='/Users/vkolhatk/Data/Constructiveness/data/train/NYT_picks_constructive_YNACC_non_constructive.csv',
                        help="The training data output CSV containing instances for constructive and non-constructive comments.")

    parser.add_argument('--test_csv', '-te', type=str, dest='test_csv', action='store',
                        default='/Users/vkolhatk/Data/Constructiveness/data/test/SOCC_constructiveness.csv',
                        help="The test data output CSV containing instances for constructive and non-constructive comments.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    cdc = ConstructivenessDataCollector()

    # Collect training data
    positive_examples_csvs = [[args.nyt_picks_csv, 1.0], [args.commentIQ_nyt_picks_csv, 1.0]]
    negative_examples_csvs = [[args.ync_expert_path, 1.0], [args.ync_mturk_path, 0.43] ]
    cdc.collect_train_data(positive_examples_csvs, negative_examples_csvs, args.train_csv)

    # Collect test data
    test_csvs = [[args.SOCC_annotated_csv,1.0]]
    cdc.collect_test_data(test_csvs, args.test_csv)