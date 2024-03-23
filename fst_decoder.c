#include <stdio.h>
#include <float.h>
#include <string.h>
#include "fst_nth.h"
#include "fst_decoder.h"

#define INT_MAX (0x3FFFFFFF)

//static const int beam = 20;
//static const float beam_delta = 0.5;
static const int beam = 20*32768;    // 20@Q15
static const int beam_delta = 16384; // 0.5@Q15
static const int min_active = 20;    // min active number
static const int max_active = 1000;  // max active number
static const float acoustic_scale = 1; // acoustic cost weight


#if 0
inline 
int partition(int32_t* a, int low, int high) {
  int i = low;
  int j = high;
  int32_t key = a[low];
  int32_t temp;
  while (i < j) {
    while(i < j && a[j] >= key) j--;
    while(i < j && a[i] <= key) i++;
    if (i < j) {
      temp = a[i];
      a[i] = a[j];
      a[j] = temp;
    }
  }
  a[low] = a[i];
  a[i] = key;
  return i;
}

inline
int32_t nth(int32_t *a, int low, int high, int n) {
  int mid = partition(a, low, high);
  if (mid == n) return a[mid];
  return
    (mid < n) ?
    nth(a, mid+1, high, n) :
    nth(a, low, mid-1, n);
}

#endif

void decoder_copy_toks(Decoder *dest, Decoder *src, Fsts* fsts) {
  int i;
  for (i = 0; i < fsts->num_states; i++) {
    token_copy(&dest->cur_toks[i], &src->cur_toks[i]);
  }
  dest->decoded_frames = src->decoded_frames;
}

void decoder_reset(Decoder *decoder, Fsts* fsts) {
  int i;
#if 1
  for (i = 0; i < fsts->num_states; i++) {
    token_reset(&decoder->pre_toks[i]);
    token_reset(&decoder->cur_toks[i]);
  }    
  decoder->cur_toks[0].active = 1;
  decoder->state_ids_len  = 0;
  decoder->decoded_frames = 0;
#endif
}

int get_cutoff(Decoder *decoder, int32_t *adaptive_beam, int32_t *best_state, Fsts* fsts) {
  int i;
  int32_t best_cost = INT_MAX;
  int cutoff_len = 0;
  // find now frame best, min cost token 
  for (i = 0; i < fsts->num_states; i++) {
    if (decoder->pre_toks[i].active) {
      int w = decoder->pre_toks[i].cost;
      decoder->cutoff[cutoff_len++] = w;
      if (w < best_cost) {
        best_cost = w;
        *best_state = i;
      }
    }
  }

  // according bset token set pruning up limit, beam_cutoff = best_cost + beam
  // set max active node number, max_active_cutoff
  // set min active node number, min_active_cutoff

  int32_t beam_cutoff = best_cost + beam,
        min_active_cutoff = INT_MAX,
        max_active_cutoff = INT_MAX;
  if (cutoff_len > max_active) {
    max_active_cutoff = nth(decoder->cutoff, 0, cutoff_len-1, max_active);
  }
  if (max_active_cutoff < beam_cutoff) {
      if (adaptive_beam)
        *adaptive_beam = max_active_cutoff - best_cost + beam_delta;
      return max_active_cutoff;
  }
  if (cutoff_len > min_active) {
    if (min_active == 0) min_active_cutoff = best_cost;
    else {
      min_active_cutoff = nth(decoder->cutoff, 0,
                              cutoff_len > max_active ?
                              max_active-1 : cutoff_len-1,
                              min_active);
    }
  }
  if (min_active_cutoff > beam_cutoff) {
    if (adaptive_beam) {
      *adaptive_beam = min_active_cutoff - best_cost + beam_delta;
    }
    return min_active_cutoff;
  } else {
    *adaptive_beam = beam;
    return beam_cutoff;
  } 
}

int process_emitting(Decoder *decoder, int16_t *likes, Fsts* fsts) {
  int i, j;
  int32_t adaptive_beam;
  int best_state = -1;
  // compute pruning up limit/threshold value：weight_cutoff
  int32_t weight_cutoff = get_cutoff(decoder, &adaptive_beam, &best_state, fsts);
  // compute expand pruning up limit/threshold value：do best token expand next frame, compute each transferred arc new cost,combine beam threshold, get next extended pruning upper limit, next_weight_cutoff
  int32_t next_weight_cutoff = INT_MAX;
  if (best_state >= 0) {
    int sid = best_state;
    const Arc* arcs = fsts->Arcs[sid];
    for (i = 0; i < fsts->num_arcs[sid]; i++) {
      Arc arc = arcs[i];
      if (arc.ilabel != 0) {
        int32_t ac_cost = -(likes[arc.ilabel-1]<<6);
        int32_t new_weight = arc.weight + decoder->pre_toks[sid].cost + ac_cost;
        if (new_weight + adaptive_beam < next_weight_cutoff) { 
          next_weight_cutoff = new_weight + adaptive_beam;
        }
      }
    }
  }

  //traversal s_id, state_id
  for (i = 0; i < fsts->num_states; i++) {
    Token *tok = &decoder->pre_toks[i];
    //First round of pruning：prune all tokens in the current frame, suppress a token that costs more than weight_cutoff
    if (!tok->active || tok->cost >= weight_cutoff) continue;
    // connection arc of traversal state
    const Arc *arcs = fsts->Arcs[i];
    for (j = 0; j < fsts->num_arcs[i]; j++) {
      Arc arc = arcs[j];
      if (arc.ilabel != 0) {
        int32_t ac_cost = -(likes[arc.ilabel-1]<<6);
        int32_t new_weight = arc.weight + tok->cost + ac_cost;
        //Second round of pruning：tokens other than the optimal token for the current frame, updated cost is calculated based on the subsequent transfer arc, if the upper limit of spreading pruning is exceeded, the transfer arc is no longer spreading
        if (new_weight < next_weight_cutoff) {
          //continue to update the expanded pruning upper/threshold, It's a constant process of estimation
          if (new_weight + adaptive_beam < next_weight_cutoff) {
            next_weight_cutoff = new_weight + adaptive_beam;
          }
          //Generate a new token
          Token *new_tok = &decoder->cur_toks[arc.nextstate];
          if (!new_tok->active || new_tok->cost > new_weight) {
            token_copy(new_tok, tok);
            new_tok->cost = new_weight;
            // skip <eps> and <sil>
            if (arc.olabel > 0 && arc.olabel != fsts->sil_index) {
              new_tok->olabel = arc.olabel;
            }
            // the phoneme corresponding to the current transition arc is not <eps>
            if (arc.phone_id > 0) {
              if (arc.phone_id == new_tok->phone_id) {
                //if the token's phoneme is the same as the arc phoneme, the count +1
                new_tok->phone_count++;
              } else {
                if (new_tok->phone_id != 0) {
                  new_tok->phone_length++;
                  if (new_tok->phone_length < MAX_PHONE_FRAMES_LEN) {
                    new_tok->phone_frames[new_tok->phone_length] = new_tok->phone_count;
                  }
                  else {
                    for (int k = 1; k < MAX_PHONE_FRAMES_LEN; k++) {
                      new_tok->phone_frames[k-1] = new_tok->phone_frames[k];
                    }
                    new_tok->phone_frames[MAX_PHONE_FRAMES_LEN-1] = new_tok->phone_count;
                  }
                }
                new_tok->phone_id = arc.phone_id;
                new_tok->phone_count = 1;
              }
            }
          } 
        } 
      }
    }
  }
  return next_weight_cutoff;
}

void process_nonemitting(Decoder *decoder, int32_t cutoff, Fsts* fsts) {
  int i, j;
  decoder->state_ids_len = 0;
  for (i = 0; i < fsts->num_states; i++) {
    if (decoder->cur_toks[i].active) {
      decoder->state_ids[decoder->state_ids_len++] = i;
    }
  }
  while (decoder->state_ids_len > 0) {
    int sid = decoder->state_ids[--decoder->state_ids_len];
    Token *tok = &decoder->cur_toks[sid];
    if (!tok->active || tok->cost >= cutoff) continue; 
    const Arc *arcs = fsts->Arcs[sid];
    for (i = 0; i < fsts->num_arcs[sid]; i++) {
      Arc arc = arcs[i];
      // deal with non-emission transfer arcs
      if (arc.ilabel == 0) {
        //updata total cost
        int32_t new_cost = tok->cost + arc.weight;
        if (new_cost < cutoff) {
          Token *new_tok = &decoder->cur_toks[arc.nextstate];
          if (!new_tok->active || new_tok->cost > new_cost) {
            token_copy(new_tok, tok);
            new_tok->cost = new_cost;
            // skip <eps> and <sil>
            if (arc.olabel > 0 && arc.olabel != fsts->sil_index) {
              new_tok->olabel = arc.olabel;
            }
            decoder->state_ids[decoder->state_ids_len++] = arc.nextstate;
          }
        }
      }
    }
  }
}

extern int frame_index_to_print;
void decoder_decode(Decoder *decoder, int16_t *likes, Fsts* fsts) {
  int i;
  for (i = 0; i < fsts->num_states; i++) {
    /*copy now token to pretoken*/
   token_copy(&decoder->pre_toks[i], &decoder->cur_toks[i]);
    /*reset now token*/
   token_reset(&decoder->cur_toks[i]);
  }
  int32_t weight_cutoff = process_emitting(decoder, likes, fsts);
#if 1
  process_nonemitting(decoder, weight_cutoff, fsts);
  decoder->decoded_frames++;
#endif
}

#if 0
int fst_is_final(int sid) {
  int i;
  for (i = 0; i < NUM_FINALS; i++) {
    if (sid == final_states[i]) return 1; 
  }
  return 0;
}
#endif
int decoder_get_result(Decoder *decoder, int32_t *olablel, int32_t *score, Fsts* fsts) {
  int i;
  int best_sid = -1;
  int32_t best_cost = INT_MAX;
  for (i = 0; i < fsts->num_finals; i++){
    int final_sid = fsts->final_states[i];
    Token *tok  = &decoder->cur_toks[final_sid];
    if(tok->active && tok->cost < best_cost){
      best_cost = tok->cost;
      best_sid = final_sid;
    }
  }
#if 0
  for (i = 0; i < fsts->num_states; i++) {
    Token *tok = &decoder->cur_toks[i];
    if (tok->active && fst_is_final(i) && tok->cost < best_cost) {
      best_cost = tok->cost;
      best_sid = i;
    }
  }
#endif
  if (best_sid == -1) {
    return 0;
  }
  Token *best_tok = &decoder->cur_toks[best_sid];
  if (best_tok->olabel == 0 || best_tok->olabel == fsts->sil_index ) {
    return 0;
  }

#if 0
  int word_frames = 0;
  for (i = 1; i < best_tok->phone_length; i += 2) {
    word_frames = best_tok->phone_frames[i] + best_tok->phone_frames[i+1];
    if (word_frames <= 2) break;
  } 
  if (word_frames <= 2) {
    return 0;
  }
#else
  int w1 = 0, w2 = 0;
  int k = min(MAX_PHONE_FRAMES_LEN, best_tok->phone_length);
  w1 = best_tok->phone_frames[k-1] + best_tok->phone_frames[k-2];
  w2 = best_tok->phone_frames[k-3] + best_tok->phone_frames[k-4];
  if (w1 <= 2 || w2 <= 2) {
    // printf("3\n");
    return 0;
  }
#endif

  *olablel = best_tok->olabel - 1;
  //*text = fsts->words[best_tok->olabel];

  //if (strcmp(*text, "preWord") == 0) return 0;

  int32_t total_cost = (-best_tok->cost);
  if (decoder->decoded_frames > 0) {
    *score = total_cost / decoder->decoded_frames;
  } else {
    *score = 0;
  }

  return 1;
}
