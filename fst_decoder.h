
#ifndef ASR_DECODER_DECODER_H_
#define ASR_DECODER_DECODER_H_
#include <stdint.h>
#include "fst_token.h"
#include "fst_def.h"
#define MAX_STATES_NUM 400
typedef struct
{
    uint16_t decoded_frames;
    uint16_t state_ids_len;
    uint16_t state_ids[MAX_STATES_NUM];
    int32_t cutoff[MAX_STATES_NUM];
    Token pre_toks[MAX_STATES_NUM];
    Token cur_toks[MAX_STATES_NUM];
} Decoder;

void decoder_reset(Decoder *decoder, Fsts* fsts);
void decoder_decode(Decoder *decoder,int16_t*likes, Fsts* fsts);
int decoder_get_result(Decoder *decoder, int32_t *olablel, int32_t *score, Fsts* fsts);
void decoder_copy_toks(Decoder *dest, Decoder *src, Fsts* fsts);

#endif

