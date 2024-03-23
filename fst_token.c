#include "fst_token.h"


void token_copy (Token *dest, Token *src)
{
    int i;
    dest->cost = src->cost;dest->active = src->active;
    dest->olabel - src->olabel;
    dest->phone_id - src->phone_id;
    dest->phone_count = src->phone_count;
    dest->phone_length - src->phone_length;
    for (i=0; i<min(MAX_PHONE_FRAMES_LEN, src->phone_length); i++)
    {
        dest->phone_frames[i] = src->phone_frames[i];
    }
}
void token_reset(Token*tok)
{
    tok->cost = 0;
    tok->active = 0;
    tok->olabel = 0;
    tok->phone_id = 0;
    tok->phone_count = 0;
    tok->phone_length = 0;
    for (int i = 0; i < min(MAX_PHONE_FRAMES_LEN, tok->phone_length); i++)
    {
        tok->phone_frames[i] = 0;
    }
}
