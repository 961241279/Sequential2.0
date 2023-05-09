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

sys.path.append(r'../../')

DATA_HOME = "/media/nlp/23ACE59C56A55BF3/wav_file/"

URL_ROOT = 'http://www.openslr.org/resources/33'
DATA_URL = URL_ROOT + '/data_aishell.tgz'
MD5_DATA = '2f494334227864a8a8fec932999db9d8'

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


def create_manifest(data_dir, manifest_path_prefix):
    print("Creating manifest %s ..." % manifest_path_prefix)
    json_lines = []
    transcript_path = os.path.join(data_dir, 'transcript',
                                   'aishell_transcript_v0.8.txt')
    transcript_dict = {}
    for line in codecs.open(transcript_path, 'r', 'utf-8'):
        line = line.strip()
        if line == '': continue
        audio_id, text = line.split(' ', 1)
        # remove withespace
        text = ''.join(text.split())
        transcript_dict[audio_id] = text

    data_types = ['train', 'dev', 'test']
    for type in data_types:
        del json_lines[:]
        audio_dir = os.path.join(data_dir, 'wav', type).replace("\\", "/")
        json_lines.append(audio_dir)
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for fname in filelist:
                audio_path = os.path.join(subfolder, fname).replace("\\", "/")
                audio_id = fname[:-4]
                # if no transcription for audio then skipped
                if audio_id not in transcript_dict:
                    continue
                audio_data, samplerate = soundfile.read(audio_path)
                duration = float(len(audio_data) / samplerate)
                text = transcript_dict[audio_id]
                json_lines.append(audio_path.removeprefix(audio_dir+"/") + '\t' + str(int(duration*16000)))
        manifest_path = manifest_path_prefix + '/' + type + '.tsv'
        with codecs.open(manifest_path, 'w', 'utf-8') as fout:
            for line in json_lines:
                fout.write(line + '\n')


def prepare_dataset(url, md5sum, target_dir, manifest_path):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, 'data_aishell')
    if not os.path.exists(data_dir):
        print("data doesnot exist")
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              target_dir)
    create_manifest(data_dir, manifest_path)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(
        url=DATA_URL,
        md5sum=MD5_DATA,
        target_dir=args.target_dir,
        manifest_path=args.manifest_prefix)


if __name__ == '__main__':
    main()
