#ifndef FST_DEF_H_
#define FST_DEF_H_

#include <stdint.h>

#define WIN_DECODE

typedef struct
{
    uint16_t nextstate;
    uint16_t ilabel;
    uint16_t olabel;
    uint16_t phone_id;
    int32_t weight;
} Arc;

// Definition for FST
typedef struct
{
    Arc** Arcs;
    int16_t* num_arcs;
    int16_t* final_states;
    int32_t num_states;
    int32_t num_finals;
    int32_t sil_index; // specify the sil index in the olabel
} Fsts;

// Definition for FST data packing
//
// 32bit data bus will be used by board
// so CHNANGE the pointer to int32 here for packing
// and store pack offset here only
// the user should restore it to the pointer before use it.
// for 32bit application, the new definition can be used directory
// for 64bit application, should transfer it to the correct one
typedef struct
{
    int32_t Arcs_offset;        //offset of fst_arcs[] in the pack data,
                                // fst_arcs[](in 32bits) in the pack data stores offset of the real arcN.
    int32_t num_arcs_offset;    // offset of num_arcs[] in the pack data
    int32_t final_states_offset; // offset of final_states[] in the pack data
    int32_t num_states;
    int32_t num_finals;
    int32_t sil_index;          // specify the sil index in the olabel
} FstsPack;

#endif
