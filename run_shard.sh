#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0, 1"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5
# data
data_url=www.openslr.org/resources/12
# use your own data path
datadir=/database
# wav data dir
wave_data=data
# Optional train_config
# 1. conf/train_transformer_large.yaml: Standard transformer
train_config=conf/train_u2++_conformer_8k.yaml  #conf/train_conformer.yaml
checkpoint= #exp/sp_spec_aug_8k/27.pt
cmvn=true
do_delta=false

dir=exp/sp_spec_aug_8k
shards_dir=/data/wenet_train_data_shard

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
# maybe you can try to adjust it if you can not get close results as README.md
average_num=10
decode_modes="attention_rescoring ctc_greedy_search ctc_prefix_beam_search attention"

. tools/parse_options.sh || exit 1;

# bpemode (unigram or bpe)
nbpe=2500
bpemode=unigram

set -e
set -u
set -o pipefail

train_set=train_8k
dev_set=dev_8k
recog_set="eval_***"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ### Task dependent. You have to design training and dev sets by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "stage 1: Feature Generation"
  
#  tools/compute_cmvn_stats.py --num_workers 24 --train_config $train_config \
#    --in_scp $wave_data/$train_set/wav.scp \
#    --out_cmvn $wave_data/$train_set/global_cmvn

fi


dict=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 2: Dictionary and Json Data Preparation"
  mkdir -p data/lang_char/

  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1

  # we borrowed these code and scripts which are related bpe from ESPnet.
  cut -f 2- -d" " $wave_data/${train_set}/text > $wave_data/lang_char/input.txt
  tools/spm_train --input=$wave_data/lang_char/input.txt --user_defined_symbols='[noise]' --character_coverage=1.0 --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
  tools/spm_encode --model=${bpemodel}.model --output_format=piece < $wave_data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict # <eos>
  wc -l ${dict}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Compute cmvn"
  # Here we use all the training data, you can sample some some data to save time
  # BUG!!! We should use the segmented data for CMVN

#  cmvn -> espnet

#  python tools/compute_cmvn_stats.py --num_workers 24 --train_config $train_config \
#    --in_scp $wave_data/$train_set/wav.segment.scp \
#    --out_cmvn $wave_data/$train_set/global_cmvn_shard

  echo "Making shards, please wait..."
  
  for x in ${train_set} ${dev_set}; do
    dst=$shards_dir/$x
    mkdir -p $dst
    tools/make_shard_list.py --resample 8000 --num_utts_per_shard 1000 \
      --num_threads 16 \
      data/$x/wav.scp data/$x/text \
      $(realpath $dst) data/$x/data.shard.list
  done

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p $dir
  INIT_FILE=$dir/ddp_init
  rm -f $INIT_FILE # delete old one before starting
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  cmvn_opts=
  $cmvn && cmvn_opts="--cmvn $wave_data/${train_set}/global_cmvn"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    python wenet/bin/train.py --gpu $gpu_id \
      --config $train_config \
      --data_type "shard" \
      --symbol_table $dict \
      --bpe_model ${bpemodel}.model \
      --train_data $wave_data/$train_set/data.shard.list \
      --cv_data $wave_data/$dev_set/data.shard.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $num_gpus \
      --ddp.rank $i \
      --ddp.dist_backend $dist_backend \
      --num_workers 8 \
      $cmvn_opts \
      --pin_memory &> $dir/train.log
  } &
  done
  wait
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  cmvn_opts=
  $cmvn && cmvn_opts="--cmvn data/${train_set}/global_cmvn"
  # TODO, Add model average here
  mkdir -p $dir/test
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=
  ctc_weight=0.5
  # Polling GPU id begin with index 0
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  idx=0
  for test in $recog_set; do
    for mode in ${decode_modes}; do
    {
      {
        test_dir=$dir/${test}_${mode}
        mkdir -p $test_dir
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
        python wenet/bin/recognize.py --gpu $gpu_id \
          --mode $mode \
          --config $dir/train.yaml \
          --data_type raw \
          --dict $dict \
          --bpe_model ${bpemodel}.model \
          --test_data $wave_data/$test/data.list \
          --checkpoint $decode_checkpoint \
          --beam_size 10 \
          --batch_size 1 \
          --penalty 0.0 \
          --result_file $test_dir/text_bpe \
          --ctc_weight $ctc_weight \
          ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

        cut -f2- -d " " $test_dir/text_bpe > $test_dir/text_bpe_value_tmp
        cut -f1 -d " " $test_dir/text_bpe > $test_dir/text_bpe_key_tmp
        tools/spm_decode --model=${bpemodel}.model --input_format=piece \
          < $test_dir/text_bpe_value_tmp | sed -e "s/▁/ /g" > $test_dir/text_value_tmp
        paste -d " " $test_dir/text_bpe_key_tmp $test_dir/text_value_tmp > $test_dir/text

        python tools/compute-wer.py --char=1 --v=1 \
          $wave_data/$test/text $test_dir/text > $test_dir/wer
      } &

      ((idx+=1))
      if [ $idx -eq $num_gpus ]; then
        idx=0
      fi
    }
    done
  done
  wait

fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip
fi

# Optionally, you can add LM and test it with runtime.
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  lm=data/local/lm
  lexicon=data/local/dict/lexicon.txt
  mkdir -p $lm
  mkdir -p data/local/dict

  # 7.1 Download & format LM
  which_lm=3-gram.pruned.1e-7.arpa.gz
  if [ ! -e ${lm}/${which_lm} ]; then
    wget http://www.openslr.org/resources/11/${which_lm} -P ${lm}
  fi
  echo "unzip lm($which_lm)..."
  gunzip -k ${lm}/${which_lm} -c > ${lm}/lm.arpa
  echo "Lm saved as ${lm}/lm.arpa"

  # 7.2 Prepare dict
  unit_file=$dict
  bpemodel=$bpemodel
  # use $dir/words.txt (unit_file) and $dir/train_960_unigram5000 (bpemodel)
  # if you download pretrained librispeech conformer model
  cp $unit_file data/local/dict/units.txt
  if [ ! -e ${lm}/librispeech-lexicon.txt ]; then
    wget http://www.openslr.org/resources/11/librispeech-lexicon.txt -P ${lm}
  fi
  echo "build lexicon..."
  tools/fst/prepare_dict.py $unit_file ${lm}/librispeech-lexicon.txt \
    $lexicon $bpemodel.model
  echo "lexicon saved as '$lexicon'"

  # 7.3 Build decoding TLG
  tools/fst/compile_lexicon_token_fst.sh \
     data/local/dict data/local/tmp data/local/lang
  tools/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;

  # 7.4 Decoding with runtime
  fst_dir=data/lang_test
  for test in ${recog_set}; do
    ./tools/decode.sh --nj 6 \
      --beam 10.0 --lattice_beam 5 --max_active 7000 --blank_skip_thresh 0.98 \
      --ctc_weight 0.5 --rescoring_weight 1.0 --acoustic_scale 1.2 \
      --fst_path $fst_dir/TLG.fst \
      --dict_path $fst_dir/words.txt \
      data/$test/wav.scp data/$test/text $dir/final.zip $fst_dir/units.txt \
      $dir/lm_with_runtime_${test}
    tail $dir/lm_with_runtime_${test}/cer
  done
fi

