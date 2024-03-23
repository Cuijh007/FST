#ifndef ASR_DECODER__TOKEN_H_
#define ASR_DECODER__TOKEN_H_
#include <stdint.h>

#define MAX_PHONE_FRAMES_LEN 20
#define min(X,Y) ((X) < (Y) ? (X) : (Y))

typedef struct
{
    int32_t cost;
    uint8_t active;
    uint8_t olabel;
    uint8_t phone_id;
    uint8_t phone_count;
    uint8_t phone_length;
    uint8_t phone_frames[MAX_PHONE_FRAMES_LEN];
}Token;

void token_copy(Token *dest, Token *src);
void token_reset(Token *tok);
#endif
