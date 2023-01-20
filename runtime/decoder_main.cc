// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iomanip>
#include <thread>
#include <utility>

#include "decoder/params.h"
#include "frontend/wav.h"
#include "utils/flags.h"
#include "utils/string.h"
#include "utils/thread_pool.h"
#include "utils/timer.h"
#include "utils/utils.h"
#include "decoder/torch_asr_model.h"

#include <unistd.h>
#include <iostream>

DEFINE_bool(simulate_streaming, false, "simulate streaming input");
DEFINE_bool(output_nbest, false, "output n-best of decode result");
DEFINE_string(wav_path, "", "single wave path");
DEFINE_string(wav_scp, "", "input wav scp");
DEFINE_string(warm_wav_scp, "", "input warmup wav scp"); // added
DEFINE_string(result, "", "result output file");
DEFINE_bool(continuous_decoding, false, "continuous decoding mode");
DEFINE_int32(thread_num, 1, "num of decode thread");
DEFINE_int32(warmup, 0, "num of warmup decode, 0 means no warmup");

std::shared_ptr<wenet::DecodeOptions> g_decode_config;
std::shared_ptr<wenet::FeaturePipelineConfig> g_feature_config;
std::shared_ptr<wenet::DecodeResource> g_decode_resource;

std::ofstream g_result;
std::mutex g_mutex;
int g_total_waves_dur = 0;
int g_total_decode_time = 0;


void decode(std::pair<std::string, std::string> wav, bool warmup = false, bool log = false) {
  int copy_time = 0;
  wenet::Timer copy_timer;

  wenet::WavReader wav_reader(wav.second, log);

  int load_time = copy_timer.Elapsed();
  copy_time += load_time;
  if (log) {
    LOG(INFO) << "WavReader time " << copy_time << "ms.";
  }
  int num_samples = wav_reader.num_samples();
  CHECK_EQ(wav_reader.sample_rate(), FLAGS_sample_rate);

  auto feature_pipeline =
      std::make_shared<wenet::FeaturePipeline>(*g_feature_config);
  feature_pipeline->AcceptWaveform(wav_reader.data(), num_samples);
  feature_pipeline->set_input_finished();
  if (log) {
    LOG(INFO) << "num frames " << feature_pipeline->num_frames();
  }
  wenet::AsrDecoder decoder(feature_pipeline, g_decode_resource,
                            *g_decode_config);

  //wenet::TorchAsrModel test_warm;
  //test_warm.WarmingUp(10000);

  int wave_dur = static_cast<int>(static_cast<float>(num_samples) /
                                  wav_reader.sample_rate() * 1000);
  int decode_time = 0;
  std::string final_result;
  while (true) {
    wenet::Timer timer;
    wenet::DecodeState state = decoder.Decode(log);
    if (state == wenet::DecodeState::kEndFeats) {
      decoder.Rescoring(log);
    }
    int chunk_decode_time = timer.Elapsed();
    decode_time += chunk_decode_time;
    if (decoder.DecodedSomething()) {
      if (log) {
        LOG(INFO) << "Partial result: " << decoder.result()[0].sentence;
      }
    }

    if (FLAGS_continuous_decoding && state == wenet::DecodeState::kEndpoint) {
      if (decoder.DecodedSomething()) {
        decoder.Rescoring(log);
        if (log) {
          LOG(INFO) << "Final result (continuous decoding): "
                    << decoder.result()[0].sentence;
        }
        final_result.append(decoder.result()[0].sentence);
      }
      decoder.ResetContinuousDecoding();
    }

    if (state == wenet::DecodeState::kEndFeats) {
      break;
    } else if (FLAGS_chunk_size > 0 && FLAGS_simulate_streaming) {
      float frame_shift_in_ms =
          static_cast<float>(g_feature_config->frame_shift) /
          wav_reader.sample_rate() * 1000;
      auto wait_time =
          decoder.num_frames_in_current_chunk() * frame_shift_in_ms -
          chunk_decode_time;
      if (wait_time > 0) {
        if (log) {
          LOG(INFO) << "Simulate streaming, waiting for " << wait_time << "ms";
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(wait_time)));
      }
    }
  }
  if (decoder.DecodedSomething()) {
    final_result.append(decoder.result()[0].sentence);
  }
  if (log) {
    LOG(INFO) << wav.first << " Final result: " << final_result << std::endl;
    LOG(INFO) << "Decoded " << wave_dur << "ms audio taken " << decode_time
            << "ms.";
  }
  if (!warmup) {
    g_mutex.lock();
    std::ostream& buffer = FLAGS_result.empty() ? std::cout : g_result;
    if (!FLAGS_output_nbest) {
      buffer << wav.first << " " << final_result << std::endl;
    } else {
      buffer << "wav " << wav.first << std::endl;
      auto& results = decoder.result();
      for (auto& r : results) {
        if (r.sentence.empty()) continue;
        buffer << "candidate " << r.score << " " << r.sentence << std::endl;
      }
    }
    g_total_waves_dur += wave_dur;
    g_total_decode_time += decode_time;
    g_mutex.unlock();
  }
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  g_decode_config = wenet::InitDecodeOptionsFromFlags();
  g_feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  g_decode_resource = wenet::InitDecodeResourceFromFlags();

  if (FLAGS_wav_path.empty() && FLAGS_wav_scp.empty()) {
    LOG(FATAL) << "Please provide the wave path or the wav scp.";
  }
  std::vector<std::pair<std::string, std::string>> waves;
  std::vector<std::pair<std::string, std::string>> warm_waves;
  if (!FLAGS_wav_path.empty()) {
    waves.emplace_back(make_pair("test", FLAGS_wav_path));
  } else {
    std::ifstream wav_scp(FLAGS_wav_scp);
    std::string line;
    while (getline(wav_scp, line)) {
      std::vector<std::string> strs;
      wenet::SplitString(line, &strs);
      CHECK_GE(strs.size(), 2);
      waves.emplace_back(make_pair(strs[0], strs[1]));
    }

    std::ifstream warm_wav_scp(FLAGS_warm_wav_scp);
    std::string warm_line;
    while (getline(warm_wav_scp, warm_line)) {
      std::vector<std::string> warm_strs;
      wenet::SplitString(warm_line, &warm_strs);
      CHECK_GE(warm_strs.size(), 2);
      warm_waves.emplace_back(make_pair(warm_strs[0], warm_strs[1]));
    }  

    if (waves.empty()) {
      LOG(FATAL) << "Please provide non-empty wav scp.";
    }
  }

  if (!FLAGS_result.empty()) {
    g_result.open(FLAGS_result, std::ios::out);
  }

  // Warmup
  if (FLAGS_warmup > 0) {
    LOG(INFO) << "Warming up...";
    {
      ThreadPool pool(FLAGS_thread_num);
      auto wav = waves[0];
      for (int i = 0; i < FLAGS_warmup; i++) {
        pool.enqueue(decode, wav, true, false); //true
      }
    }
    LOG(INFO) << "Warmup done.";
  }

  {
    ThreadPool pool(FLAGS_thread_num);
    wenet::Timer warm_up_check;
    //auto wav_test = waves[1];
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < warm_waves.size(); j++) {
        auto wav_test = warm_waves[j];
        decode(wav_test, true, false);
      }
    }
    float warm_up_time = warm_up_check.Elapsed();
    LOG(INFO) << "warming up time " << (warm_up_time / 1000.) << "seconds";
    for (auto& wav : waves) {
      //warm_up_check.Reset();
      //LOG(INFO) << "warming up ..";
      //for (int i = 0; i < 5; i++) {
        //pool.enqueue(decode, wav, true, false);
      //  decode(wav, true, false);
      //}

      //int warm_up_time = warm_up_check.Elapsed();
      //std::cout << "warming up time " << warm_up_time << "ms." << std::endl;
      //LOG(INFO) << "done.";
      //pool.enqueue(decode, wav, false, true);
      decode(wav, false, true);
    }
  }

  LOG(INFO) << "Total: decoded " << g_total_waves_dur << "ms audio taken "
            << g_total_decode_time << "ms.";
  LOG(INFO) << "RTF: " << std::setprecision(4)
            << static_cast<float>(g_total_decode_time) / g_total_waves_dur;
  return 0;
}
