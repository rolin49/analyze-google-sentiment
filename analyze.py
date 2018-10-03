import argparse, json, os.path, sys
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import numpy as np

def calculate_goog_score(sent):
    if sent > 0:
        return 0
    elif sent < 0:
        return 1
    else:
        return 2

def calculate_goog_score_word(score):
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

def make_human_eval_filename(transcript_filename):
    human_eval_filename_root, human_eval_filename_ext = os.path.splitext(
        transcript_filename)
    return human_eval_filename_root + '-userdata' + human_eval_filename_ext

def write_to_file(sents, total, filename):
    print('Saving data to {}....'.format(filename))
    sents["total"] = total
    with open(filename, 'w') as file:
        json.dump(sents, file)
    print('Done!')

def calculate_total_per_speaker(sentiments):
    total = 0
    for human in range(3):
        for google in range(3):
            total += sentiments[human][google]
    return total

def print_overall_accuracy(sentiments, total):
    accurate_ct = 0
    for speaker in sentiments:
        for score in range(3):
            accurate_ct += sentiments[speaker][score][score]
    print('Overall accuracy: {}%'.format(100 * accurate_ct / total))

def print_accurate_stats(sentiments, total):
    for sent in range(3):
        sents = ['posi', 'neg', 'neut']
        n_accurate = sentiments[sent][sent]
        if sum(sentiments[sent]) == 0:
            print('No {} statements found!'.format(sents[sent]))
        else:
            print('Accurate ID of {}% of {} stmts, {}% of speaker total'
                  .format(100 * n_accurate / sum(sentiments[sent]),
                          sents[sent],
                          100 * n_accurate / total))

def print_inaccurate_stats(sentiments, total):
    sents = ['posi', 'neg', 'neut']
    for human in range(3):
        for google in range(3):
            if human != google:
                print('Incorrect ID of {} {} stmts as {}: {}% of speaker total'
                      .format(sentiments[human][google],
                              sents[human],
                              sents[google],
                              100 * sentiments[human][google] / total))

def print_percent_correct_of_sents_labeled_smt(sentiments, total):
    sents = ['posi', 'neg', 'neut']
    for sent in range(3):
        if sum(row[sent] for row in sentiments) == 0:
            print('No {} statements found!'.format(sents[sent]))
        else:
            print('{0}% of stmts Google thought were {1} were actually {1}'
                  .format(100 * sentiments[sent][sent] /
                          sum(row[sent] for row in sentiments),
                          sents[sent]))

def print_overall_speaker_accuracy(sentiments, speaker_total):
    accurate = 0
    for score in range(3):
        accurate += sentiments[score][score]
    print('Overall speaker accuracy: {}%'
          .format(100 * accurate / speaker_total))

def print_speaker_accuracy(sentiments):
    for speaker in sentiments:
        print('Speaker {}:'.format(speaker))
        speaker_total = calculate_total_per_speaker(sentiments[speaker])
        print_overall_speaker_accuracy(sentiments[speaker], speaker_total)
        print_accurate_stats(sentiments[speaker], speaker_total)
        print_inaccurate_stats(sentiments[speaker], speaker_total)
        print_percent_correct_of_sents_labeled_smt(sentiments[speaker],
                                                   speaker_total)

def print_stats(sentiments, total):
    print_overall_accuracy(sentiments, total)
    print_speaker_accuracy(sentiments)

def plot_confusion_matrix(sentiments, filename):
    sents = ['posi', 'neg', 'neut']
    for speaker in sentiments:
        f = plt.figure()
        plt.imshow(np.array(sentiments[speaker]))
        plt.set_cmap('Reds')
        plt.colorbar()
        for x in range(3):
            for y in range(3):
                plt.annotate(str(sentiments[speaker][x][y]), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center')
        plt.suptitle("Speaker " + str(speaker))
        plt.xticks(range(3), sents)
        plt.yticks(range(3), sents)
        plt.xlabel("Google")
        plt.ylabel("Human")
        f.savefig(filename + "_speaker" + str(speaker) + ".pdf")
        plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'transcript_filename',
        help='The filename of the transcript you\'d like to analyze.')
    args = parser.parse_args()

    """Run a sentiment analysis request on text within a passed filename."""
    client = language.LanguageServiceClient()

    with open(args.transcript_filename, 'r') as transcript_file:
        content = json.load(transcript_file)

    human_eval_filename = make_human_eval_filename(args.transcript_filename)
    if os.path.isfile(human_eval_filename):
        resp = input('User data was found for {} (in {})!\n'
              'Would you like to use it? y/n: '
              .format(args.transcript_filename, human_eval_filename))
        if resp == 'y':
            sentiments = {}
            with open(human_eval_filename, 'r') as human_eval_file:
                sentiments = json.load(human_eval_file)
            total = sentiments.pop("total")
            print_stats(sentiments, total)
            plot_confusion_matrix(sentiments, args.transcript_filename)
            sys.exit(0)
        elif resp == 'n':
            pass

    print('Analyzing accuracy of Google Cloud Natural Language API....')

    total = 0

    sentiment_statistics = {}

    for para_obj in content:
        para = para_obj["transcript"]
        speaker_number = int(para_obj["speaker"])
        if speaker_number not in sentiment_statistics:
            sentiment_statistics[speaker_number] = [[0,0,0],[0,0,0],[0,0,0]]
        document = types.Document(
            content=para,
            type=enums.Document.Type.PLAIN_TEXT)
        annotations = client.analyze_sentiment(document=document)
        for index, sentence in enumerate(annotations.sentences):
            total += 1
            sentence_sentiment = sentence.sentiment.score
            goog_score = calculate_goog_score(sentence_sentiment)
            print('--> {}'.format(sentence.text.content))
            human_eval = int(input(
                    'Enter 0 for positive, 1 for negative, 2 for neutral: '))
            print('Google thought it was {}.'.format(
                    calculate_goog_score_word(
                        calculate_goog_score(sentence_sentiment))))
            sentiment_statistics[speaker_number][human_eval][goog_score] += 1

    print('Done!')
    
    print_stats(sentiment_statistics, total)

    plot_confusion_matrix(sentiment_statistics, args.transcript_filename)
    
    write_to_file(sentiment_statistics, total, human_eval_filename)
