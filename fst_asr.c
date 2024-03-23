#include "fst_decoder.h"
#include "fst_asr.h"
#include <fst_def.h>
#include <stdio.h>
#include <stdlib.h>

#define SCORE_QUANT_COEF    (1<<15)
#define LIKES_QUANT_COEF    (32768.0)



#ifdef WIN_DECODE 
#define WIN_STEP 35 
#define WIN_SHIFT1 (WIN_STEP*1) 
#define WIN_SHIFT2 (WIN_STEP*2)
#define WIN_LENGTH (WIN_STEP*3) 
Decoder *decoder_shift1 = NULL;
Decoder *decoder_shift2 = NULL;
static int start_decoder_shift1 = 0;
static int start_decoder_shift2 = 0;
static int copy_decoder_shift = 1;
#endif

#define WIN_REST 10

Decoder *decoder = NULL;
static int sample_count = 0;
static int result_count = 0;
static Fsts* fsts_local;
int flag = 0;

int fn_asr_init(Fsts* fsts) {

  fsts_local = fsts;

  return 0;
}

void fn_asr_reset() {
  sample_count = 0;
  result_count = 0;
  decoder_reset(decoder, fsts_local);
#ifdef WIN_DECODE
  decoder_reset(decoder_shift1, fsts_local);
  decoder_reset(decoder_shift2, fsts_local);
  start_decoder_shift1 = 0;
  start_decoder_shift2 = 0;
  copy_decoder_shift = 1;
#endif
}

void fst_dec_init() {
  decoder = (Decoder*)malloc(sizeof(Decoder));
#ifdef WIN_DECODE
  decoder_shift1 = (Decoder*)malloc(sizeof(Decoder));
  decoder_shift2 = (Decoder*)malloc(sizeof(Decoder));
#endif
}

int fn_asr_process(short *buf, int32_t *cmd_id, int32_t *score) {
  int i, n = 0;
  sample_count++;   // reset to zero after a cmd

#if 0
  for(i = 0; i<buf_len; i++)
  {
    likes[i] = buf[i]/LIKES_QUANT_COEF;
  }
#endif  

  decoder_decode(decoder, buf, fsts_local);

  int32_t asr_result;
  int32_t asr_score;
  int rs = decoder_get_result(decoder, &asr_result, &asr_score, fsts_local);

#ifdef WIN_DECODE 
  if (sample_count == WIN_SHIFT1 + 1) {
    start_decoder_shift1 = 1;
  }
  if (sample_count == WIN_SHIFT2 + 1) {
    start_decoder_shift2 = 1;
  }
  if (start_decoder_shift1) {
    decoder_decode(decoder_shift1, buf, fsts_local);
  }
  if (start_decoder_shift2) {
    decoder_decode(decoder_shift2, buf, fsts_local);
  }
  if (rs != 1 && sample_count > WIN_STEP) {
    if (copy_decoder_shift == 1){
        rs = decoder_get_result(decoder_shift1, &asr_result, &asr_score, fsts_local);
    } else {
        rs = decoder_get_result(decoder_shift2, &asr_result, &asr_score, fsts_local);
    }
  }
  if (sample_count == WIN_LENGTH) {
    decoder_reset(decoder, fsts_local);
    if (copy_decoder_shift == 1) {
      decoder_copy_toks(decoder, decoder_shift1, fsts_local);
      decoder_reset(decoder_shift1, fsts_local);
      copy_decoder_shift = 2;
    } else if (copy_decoder_shift == 2){
      decoder_copy_toks(decoder, decoder_shift2, fsts_local);
      decoder_reset(decoder_shift2, fsts_local);
      copy_decoder_shift = 1;
    }
    sample_count -= WIN_STEP;
  }
  if (rs == 1) {
    if (++result_count == 5) {
      *cmd_id  = asr_result;    
      *score = asr_score;
      fn_asr_reset();
      return 1;
    }
    if (sample_count == WIN_LENGTH)
      result_count = 0;
  } else {
    result_count = 0;
  }
  *cmd_id = 0;
  return 0;
#else
  if (rs == 1) {
    *cmd_id  = asr_result;    
    *score = asr_score;
    fn_asr_reset();
    return 1;
  }
  *cmd_id = 0;
  return 0; 
#endif
}

void fn_asr_release() {
}

