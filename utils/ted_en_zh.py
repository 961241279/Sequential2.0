#!/usr/bin/env python
# coding=utf-8

# author@baidu deepspeech
# modified by Pelhans

"""
Prepare Aishell mandarin dataset
Create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import codecs
import soundfile
import json
import argparse
import shutil
import random

sys.path.append(r'../../')

DATA_HOME = "/media/nlp/23ACE59C56A55BF3/wav_file/"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "aishell",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()

def split_dataset(data_dir):

    """audio_dir = os.path.join(data_dir, 'train-all')
    for _, _, train_list in os.walk(audio_dir):
        train_10h_files = []
        current_length = 0
        random.shuffle(train_list)
        for train_file in train_list:
            audio_data, samplerate = soundfile.read(os.path.join(audio_dir, train_file))
            duration = float(len(audio_data) / samplerate)
            current_length += duration
            if current_length <= 36000:
                train_10h_files.append(train_file)
            else:
                break
        for train_10h_file in train_10h_files:
            movefile(os.path.join(audio_dir, train_10h_file), "D:/ted_en_zh/train-10h/", copy=True)"""

    audio_dir =  os.path.join(data_dir, 'train-segment/')
    for _, _, file_list in os.walk(audio_dir):
        translation_dict = {}
        with open("D:/ted_en_zh/En-Zh/train.en-zh", 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 3, line
                audio_id = items[0]
                translation_dict[audio_id] = items[2]
        full_list = []
        for file_name in file_list:
            if file_name in translation_dict:
                full_list.append(file_name)
        random.shuffle(full_list)
        n_total = len(full_list)
        print(f"total files:{n_total}")
        offset1 = int(n_total * 0.01)
        offset2 = offset1 + int(n_total * 0.02)
        if n_total == 0 or offset1 < 1:
            return [], full_list
        valid_list = full_list[:offset1]
        test_list = full_list[offset1:offset2]
        train_list = full_list[offset2:]
        for vaild_file in valid_list:
            movefile(os.path.join(audio_dir,vaild_file), os.path.join(data_dir, "valid-segment/"), copy=False)
        for test_file in test_list:
            movefile(os.path.join(audio_dir, test_file), os.path.join(data_dir, "test-segment/"), copy=False)
        current_length = 0
        for data_file in train_list:
            audio_data, samplerate = soundfile.read(os.path.join(audio_dir,data_file))
            duration = float(len(audio_data) / samplerate)
            current_length += duration
            if current_length <= 36000:
                movefile(os.path.join(audio_dir,data_file), "D:/ted_en_zh/train-10h/", copy=True)
            elif current_length > 36000 and current_length <= 396000:
                movefile(os.path.join(audio_dir,data_file), "D:/ted_en_zh/train-100h/", copy=True)
            else:
                break;

def movefile(srcfile,dstpath,copy=False):                       # 移动函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        if(copy):
            shutil.copy(srcfile, os.path.join(dstpath, fname))      # 复制文件
        else:
            shutil.move(srcfile, os.path.join(dstpath, fname))      # 移动文件
        print("move %s -> %s"%(srcfile, os.path.join(dstpath, fname)))

def create_manifest(data_dir, manifest_path_prefix):
    print("Creating manifest %s ..." % manifest_path_prefix)
    json_lines = []
    """transcript_path = os.path.join(data_dir, 'En-Zh',
                                   'train.en-zh')
    transcript_dict = {}
    for line in codecs.open(transcript_path, 'r', 'utf-8'):
        line = line.strip()
        if line == '': continue
        audio_id, text = line.split(' ', 1)
        # remove withespace
        text = ''.join(text.split())
        transcript_dict[audio_id] = text"""

    data_types = ['train-10h', 'train-100h', 'valid-segment', 'test-segment']
    for type in data_types:
        del json_lines[:]
        audio_dir = os.path.join(data_dir, type).replace("\\", "/")
        json_lines.append(audio_dir)
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for fname in filelist:
                audio_path = os.path.join(subfolder, fname).replace("\\", "/")
                audio_id = fname[:-4]
                # if no transcription for audio then skipped
                """if audio_id not in transcript_dict:
                    continue
                text = transcript_dict[audio_id]"""
                audio_data, samplerate = soundfile.read(audio_path)
                duration = float(len(audio_data) / samplerate)
                audio_path = audio_path
                json_lines.append(audio_path.removeprefix(audio_dir+"/") + '\t' + str(int(duration*16000)))
        manifest_path = manifest_path_prefix + '/' + type + '.tsv'
        with codecs.open(manifest_path, 'w', 'utf-8') as fout:
            for line in json_lines:
                fout.write(line + '\n')

def create_labels(data_dir, manifest_path_prefix):
    print("Creating labels %s ..." % manifest_path_prefix)
    label_lines = []
    data_types = ['train-10h', 'train-100h', 'valid-segment', 'test-segment']
    translation_dict = {}
    with open("D:/ted_en_zh/En-Zh/train.en-zh", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 3, line
            audio_id = items[0]
            translation_dict[audio_id] = items[2]
    for type in data_types:
        del label_lines[:]
        manifest_path = manifest_path_prefix + '/' + type + '.tsv'
        wrd_path = manifest_path_prefix + '/' + type + '.wrd'
        ltr_path = manifest_path_prefix + '/' + type + '.ltr'
        bpe_path = manifest_path_prefix + '/' + type + '.bpe'
        out_of_label = 0
        with open(manifest_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            root_dir = first_line
            labels = []
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                file_name = items[0].split("/")[-1]
                if not file_name in translation_dict:
                    print(type+" "+file_name)
                    out_of_label = out_of_label + 1
                    continue
                labels.append(translation_dict[file_name])
            print(out_of_label)
            with open(wrd_path, 'w', encoding='utf-8') as fout:
                for label in labels:
                    fout.write(label + '\n')
            """with open(bpe_path, 'w', encoding='utf-8') as fout:
                fout.write("_")
                for label in labels:
                    fout.write(" _".join(line.split(" ")) + '\n')"""
            with open(ltr_path, 'w', encoding='utf-8') as fout:
                for label in labels:
                    tmp_line = " ".join(label)
                    fout.write(" | ".join(tmp_line.split("   ")) + '\n')

def create_segment(data_dir, manifest_path_prefix):
    write_translations(manifest_path_prefix)
    import io
    import sentencepiece as spm
    import pickle
    model = io.BytesIO()
    translations_path = manifest_path_prefix + "/" + "translations.txt"
    model_path = 'D:/MachineLearning/Sequential2.0/save/pretrainted/bpe.model'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        spm.SentencePieceTrainer.train(input=translations_path, model_writer=model, vocab_size=32000, character_coverage=1.0, model_type="bpe")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
    data_types = ['train-10h', 'train-100h', 'valid-segment', 'test-segment']
    for type in data_types:
        bpe_path = manifest_path_prefix + '/' + type + '.bpe'
        wrd_path = manifest_path_prefix + '/' + type + '.wrd'
        bpe_res = []
        with open(wrd_path, 'r', encoding='utf-8') as fin:
            for i,line in enumerate(fin):
                encoded = sp.encode(line, out_type=str)
                bpe_res.append(" ".join(encoded))
        with open(bpe_path, 'w', encoding='utf-8') as fout:
            for bpe_item in bpe_res:
                fout.write(bpe_item + '\n')

def write_translations(manifest_path_prefix):
    translations_path = manifest_path_prefix +"/" +"translations.txt"
    if os.path.exists(translations_path):
        return
    translations = []
    with open("D:/ted_en_zh/En-Zh/train.en-zh", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 3, line
            audio_id = items[0]
            translations.append(items[2])
    with open(translations_path, 'w', encoding='utf-8') as fout:
        for translation in translations:
            fout.write(translation + '\n')

def prepare_dataset(target_dir, manifest_path):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, 'ted_en_zh/')
    """split_dataset(data_dir)
    create_manifest(data_dir, manifest_path)"""
    """create_labels(data_dir, manifest_path)"""
    create_segment(data_dir, manifest_path)

def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(
        target_dir=args.target_dir,
        manifest_path=args.manifest_prefix)


if __name__ == '__main__':
    main()