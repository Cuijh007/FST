/*
 * This program is only used to show a portion of my work to Edinburgh (EPCC)
 * Finite State Acceptor is usually decoded after npu processing. The following is the specific decoding process.
 * For full use, phonetic dictionaries and acoustic models are required.
 *
 * High Performance Computing, this is for you!!!
 *
 * Author:   Jiahao Cui
 * UUN:      S2602811
 * Data:     25 May 2023
 * Email:    parker_cuijiahao@163.com
 * Version:  1.0
 */


#include <stdio.h>
#include <stdint.h>

#include "fst_asr.h"


#define MAX_NPU_OUT_SIZE    (1024)


int main(int argc, char** argv)
{
    int16_t word_scores[MAX_NPU_OUT_SIZE] = {0};
    int32_t cmd_ids = 0;
    int32_t cmd_scores = 0;

    fst_dec_init();
    fn_asr_process((int16_t *)word_scores, cmd_ids, (int32_t*)cmd_scores);

    return 0;
}