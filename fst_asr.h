#ifndef ASR_FN_ASR_H_
#define ASR_FN_ASR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "fst_def.h"

// Return value   : 0 - OK, -1 - Error
int  fn_asr_init(Fsts* fsts);
void fn_asr_reset();
void fst_dec_init();

/*****************************************
* Input:
*      - buf      : out of am 
*      - buf_len  : size of am out
*
* Output:
*      - cmd_id   : Index of the out command 
*      - score    : The confidence of ASR
*
* Return value    :  0 - No result
*                    1 - Has result 
*                   -1 - Error
******************************************/
int fn_asr_process(short *buf, int32_t *cmd_id, int32_t *score);
void fn_asr_release();

#ifdef __cplusplus
}
#endif

#endif
