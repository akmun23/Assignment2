// nn.cpp - cascade SELECTOR: raw 70x70 grayscale image -> 3 scores + the 8x8 feature map after POINTWISE 1, which the heads continue from.
//   image -> STEM -> DEPTHWISE 1 -> POINTWISE 1 (max-pool, = feature_map) -> DEPTHWISE 2 -> POINTWISE 2 (sum-pool) -> HIDDEN -> OUTPUT
// The class with the LOWEST score is excluded and the head of the remaining pair decides; a wrong exclusion cannot be recovered.
// Inversion (ink = 255 - pixel), the 1/255 scale and the pooling /4 are folded into the weights; all arithmetic is integer.
// run_layer runs every layer from LAYER_TABLE on the same LANES lanes (one multiplier each): a lane computes one output channel, the reduction runs in time at II=1,
// feature maps are banked per lane (channel c: bank c % LANES, group c / LANES).
#include "ap_int.h"

// ===== NETWORK CONFIGURATION (derived from the trained model; the weight tables are in nn_weights.h) =====
// Source model a6b10c12h16k4s4o3: 3615 parameters.
// weight bits {'stem': 8, 'dw1': 7, 'pw1': 8, 'dw2': 7, 'pw2': 7, 'fc1': 8, 'fc2': 7}
// activation bits {'stem': 7, 'dw1': 10, 'pw1': 9, 'dw2': 11, 'pw2': 11, 'fc1': 10}
// worst-case accumulator bits {'stem': 19, 'dw1': 16, 'pw1': 19, 'dw2': 18, 'pw2': 19, 'fc1': 25, 'fc2': 20}

// ---------- types ----------
typedef ap_int<8> weight_t;          // folded weights, widest layer
typedef ap_int<14> data_t;          // every stored activation (also the 2x2 sums and the ink = 255 - pixel)
typedef ap_int<25> accumulator_t;   // widest worst-case accumulator of any layer

// ---------- multiplier lanes (the weight ROM is banked for it; to change it re-run the exporter with --lanes N) ----------
static const int LANES = 6;

// ---------- network sizes ----------
static const int IMAGE_HEIGHT = 70;
static const int IMAGE_WIDTH = 70;
static const int STEM_KERNEL_SIZE = 4;              // stem: stride-4 convolution, 4x4 kernel (no padding) or 5x5 (1 pixel padding top and left)
static const int STEM_CHANNELS = 6;
static const int POINTWISE1_CHANNELS = 10;          // also the channels of depthwise 2
static const int POINTWISE2_CHANNELS = 12;
static const int HIDDEN_NEURONS = 16;
static const int OUTPUT_COUNT = 3;                // 3 class scores
static const int STEM_MAP_HEIGHT = 17;           // stem and depthwise 1
static const int STEM_MAP_WIDTH = 17;
static const int POOLED_MAP_HEIGHT = 8;          // after the 2x2 max-pool
static const int POOLED_MAP_WIDTH = 8;
static const int FEATURE_MAP_HEIGHT = 4;         // after the 2x2 sum-pool
static const int FEATURE_MAP_WIDTH = 4;
static const int FEATURE_VECTOR_LENGTH = POINTWISE2_CHANNELS * FEATURE_MAP_HEIGHT * FEATURE_MAP_WIDTH;   // input of the hidden layer
static const int PASSED_FEATURE_COUNT = POINTWISE1_CHANNELS * POOLED_MAP_HEIGHT * POOLED_MAP_WIDTH;   // size of the feature map handed to a head

// ---------- per-layer requantisation: activation = clamp((accumulator + 2^(shift-1)) >> shift, low, high) ----------
static const int STEM_REQUANT_SHIFT = 10;
static const int STEM_CLAMP_LOW = 0;
static const int STEM_CLAMP_HIGH = 127;
static const int DEPTHWISE1_REQUANT_SHIFT = 4;
static const int DEPTHWISE1_CLAMP_LOW = -512;
static const int DEPTHWISE1_CLAMP_HIGH = 511;
static const int POINTWISE1_REQUANT_SHIFT = 7;
static const int POINTWISE1_CLAMP_LOW = 0;
static const int POINTWISE1_CLAMP_HIGH = 511;
static const int DEPTHWISE2_REQUANT_SHIFT = 6;
static const int DEPTHWISE2_CLAMP_LOW = -1024;
static const int DEPTHWISE2_CLAMP_HIGH = 1023;
static const int POINTWISE2_REQUANT_SHIFT = 4;
static const int POINTWISE2_CLAMP_LOW = 0;
static const int POINTWISE2_CLAMP_HIGH = 2047;
static const int HIDDEN_REQUANT_SHIFT = 9;
static const int HIDDEN_CLAMP_LOW = 0;
static const int HIDDEN_CLAMP_HIGH = 1023;
static const int SCORE_SHIFT = 7;          // accumulator -> Q16.16 scores_q16 (exact left shift)

// ---------- weight / bias ROM layout (the tables themselves are in nn_weights.h) ----------
// Banked per lane: lane l holds output channels l, l+LANES, ... Layer order stem, depthwise 1, pointwise 1, depthwise 2,
// pointwise 2, hidden, output; inside a layer: [channel group][reduction index].
static const int WEIGHT_ROM_SIZE = 667;     // 4002 weights, 28008 bits of information
static const int BIAS_ROM_SIZE = 12;
static const int STEM_WEIGHT_OFFSET = 0;
static const int STEM_BIAS_OFFSET = 0;
static const int DEPTHWISE1_WEIGHT_OFFSET = 16;
static const int DEPTHWISE1_BIAS_OFFSET = 1;
static const int POINTWISE1_WEIGHT_OFFSET = 25;
static const int POINTWISE1_BIAS_OFFSET = 2;
static const int DEPTHWISE2_WEIGHT_OFFSET = 37;
static const int DEPTHWISE2_BIAS_OFFSET = 4;
static const int POINTWISE2_WEIGHT_OFFSET = 55;
static const int POINTWISE2_BIAS_OFFSET = 6;
static const int HIDDEN_WEIGHT_OFFSET = 75;
static const int HIDDEN_BIAS_OFFSET = 8;
static const int OUTPUT_WEIGHT_OFFSET = 651;
static const int OUTPUT_BIAS_OFFSET = 11;
// ===== END NETWORK CONFIGURATION =====

#include "nn_weights.h"   // only the weight and bias tables

#define CHANNEL_GROUPS(channel_count) (((channel_count) + LANES - 1) / LANES)   // lane groups needed for that many channels
#define MAXIMUM_OF(first, second) ((first) > (second) ? (first) : (second))

static const int POINTWISE1_CHANNEL_GROUPS = CHANNEL_GROUPS(POINTWISE1_CHANNELS);
static const int POINTWISE2_CHANNEL_GROUPS = CHANNEL_GROUPS(POINTWISE2_CHANNELS);
static const int HIDDEN_CHANNEL_GROUPS = CHANNEL_GROUPS(HIDDEN_NEURONS);
static const int OUTPUT_CHANNEL_GROUPS = CHANNEL_GROUPS(OUTPUT_COUNT);

static const int POOLED_MAP_SIZE = POOLED_MAP_HEIGHT * POOLED_MAP_WIDTH;        // words of one channel of the 8x8 maps
static const int FEATURE_MAP_SIZE = FEATURE_MAP_HEIGHT * FEATURE_MAP_WIDTH;     // words of one channel of the 4x4 map

static const int STEM_CHANNEL_GROUPS = CHANNEL_GROUPS(STEM_CHANNELS);
static const int STEM_MAP_SIZE = STEM_MAP_HEIGHT * STEM_MAP_WIDTH;              // words of one channel of the 17x17 maps
static const int IMAGE_WORDS = IMAGE_HEIGHT * IMAGE_WIDTH / 4;                  // 1225 32-bit words, 4 pixels each
static_assert(STEM_KERNEL_SIZE == 4 || STEM_KERNEL_SIZE == 5,
              "the stem is a stride-4 convolution with a 4x4 kernel (no padding) or a 5x5 kernel (1 pixel padding top and left)");
static const int STEM_PADDING = (STEM_KERNEL_SIZE - 3) / 2;                     // 0 for 4x4, 1 for 5x5

// Words per bank of ONE feature buffer: the largest map that is ever stored. Two such buffers share each bank
// (ping-pong): a layer reads one half and writes the other.
static const int BUFFER_WORDS_PER_BANK = MAXIMUM_OF(MAXIMUM_OF(STEM_CHANNEL_GROUPS * STEM_MAP_SIZE, POINTWISE1_CHANNEL_GROUPS * POOLED_MAP_SIZE),
                                                    MAXIMUM_OF(POINTWISE2_CHANNEL_GROUPS * FEATURE_MAP_SIZE, HIDDEN_CHANNEL_GROUPS));
static const int FIRST_HALF = 0;                       // word offset of the first ping-pong buffer inside a bank
static const int SECOND_HALF = BUFFER_WORDS_PER_BANK;  // word offset of the second one

// ============================================================
// LAYER TABLE
// ============================================================
// How the engine finds the input value of each multiplication:
enum AddressingMode {
    MODE_STEM = 0,        // pixel of an image patch (ink = 255 - pixel); the same value goes to every lane
    MODE_DEPTHWISE = 1,   // 3x3 neighbourhood of the lane's own channel (zero padded); every lane reads its own bank
    MODE_POINTWISE = 2    // input channel by input channel (also the fully connected layers); the same value goes to every lane
};
// Pooling fused into the output stage:
enum Pooling { POOL_NONE = 0, POOL_MAX = 1, POOL_SUM = 2 };
// Where a finished output channel goes:
enum StoreMode {
    STORE_ACTIVATIONS = 0,   // requantise (ReLU, pooling) and write into the feature buffer
    STORE_SCORES = 2         // the accumulator is a class score
};

struct LayerConfig {
    int addressing_mode;
    int output_positions;       // output positions per channel (pooled grid for pointwise layers)
    int output_row_width;       // width of one row of output positions
    int channel_groups;         // groups of LANES output channels
    int pool_positions;         // positions pooled into one output (4 for a 2x2 pool, otherwise 1)
    int reduction_length;       // multiplications accumulated into one output (taps x input channels)
    int taps_per_row;           // taps in one row of the reduction (stem: kernel size, depthwise: 3)
    int input_channel_shift;    // pointwise / fully connected: input channel = reduction_index >> input_channel_shift
    int input_row_width;        // width of one row of the input map (pointwise layers that pool)
    int input_group_stride;     // words between two channel groups of the input buffer
    int output_group_stride;    // words between two channel groups of the output buffer
    int pooling;
    int requant_shift;          // rounding right shift from accumulator to activation
    int clamp_low;              // lowest activation (0 = ReLU)
    int clamp_high;             // highest activation
    int weight_offset;          // where the layer starts in the weight ROM
    int bias_offset;            // where the layer starts in the bias ROM
    int source_offset;          // word offset of the input buffer half inside a bank
    int destination_offset;     // word offset of the output buffer half inside a bank
    int store_mode;
};

// Positions in LAYER_TABLE
enum LayerIndex {
    STEM_LAYER = 0, DEPTHWISE1_LAYER = 1, POINTWISE1_LAYER = 2, DEPTHWISE2_LAYER = 3,
    POINTWISE2_LAYER = 4, HIDDEN_LAYER = 5, OUTPUT_LAYER = 6, LAYER_COUNT = 7
};
static const LayerConfig LAYER_TABLE[LAYER_COUNT] = {
    {   // STEM: stride-4 convolution on the image
        MODE_STEM, STEM_MAP_SIZE, STEM_MAP_WIDTH, STEM_CHANNEL_GROUPS, 1, STEM_KERNEL_SIZE * STEM_KERNEL_SIZE, STEM_KERNEL_SIZE, 0, 0,
        0, STEM_MAP_SIZE, POOL_NONE,
        STEM_REQUANT_SHIFT, STEM_CLAMP_LOW, STEM_CLAMP_HIGH,
        STEM_WEIGHT_OFFSET, STEM_BIAS_OFFSET, SECOND_HALF, FIRST_HALF, STORE_ACTIVATIONS
    },
    {   // DEPTHWISE 1: 3x3 per channel on the 17x17 map
        MODE_DEPTHWISE, STEM_MAP_SIZE, STEM_MAP_WIDTH, STEM_CHANNEL_GROUPS, 1, 9, 3, 0, 0,
        STEM_MAP_SIZE, STEM_MAP_SIZE, POOL_NONE,
        DEPTHWISE1_REQUANT_SHIFT, DEPTHWISE1_CLAMP_LOW, DEPTHWISE1_CLAMP_HIGH,
        DEPTHWISE1_WEIGHT_OFFSET, DEPTHWISE1_BIAS_OFFSET, FIRST_HALF, SECOND_HALF, STORE_ACTIVATIONS
    },
    {   // POINTWISE 1: 1x1 across channels, ReLU, 2x2 max-pool (17x17 -> 8x8)
        MODE_POINTWISE, POOLED_MAP_SIZE, POOLED_MAP_WIDTH, POINTWISE1_CHANNEL_GROUPS, 4, STEM_CHANNELS, 1, 0, STEM_MAP_WIDTH,
        STEM_MAP_SIZE, POOLED_MAP_SIZE, POOL_MAX,
        POINTWISE1_REQUANT_SHIFT, POINTWISE1_CLAMP_LOW, POINTWISE1_CLAMP_HIGH,
        POINTWISE1_WEIGHT_OFFSET, POINTWISE1_BIAS_OFFSET, SECOND_HALF, FIRST_HALF, STORE_ACTIVATIONS
    },
    {   // DEPTHWISE 2: 3x3 per channel on the 8x8 map
        MODE_DEPTHWISE, POOLED_MAP_SIZE, POOLED_MAP_WIDTH, POINTWISE1_CHANNEL_GROUPS, 1, 9, 3, 0, 0,
        POOLED_MAP_SIZE, POOLED_MAP_SIZE, POOL_NONE,
        DEPTHWISE2_REQUANT_SHIFT, DEPTHWISE2_CLAMP_LOW, DEPTHWISE2_CLAMP_HIGH,
        DEPTHWISE2_WEIGHT_OFFSET, DEPTHWISE2_BIAS_OFFSET, FIRST_HALF, SECOND_HALF, STORE_ACTIVATIONS
    },
    {   // POINTWISE 2: 1x1 across channels, ReLU, 2x2 sum-pool (8x8 -> 4x4)
        MODE_POINTWISE, FEATURE_MAP_SIZE, FEATURE_MAP_WIDTH, POINTWISE2_CHANNEL_GROUPS, 4, POINTWISE1_CHANNELS, 1, 0, POOLED_MAP_WIDTH,
        POOLED_MAP_SIZE, FEATURE_MAP_SIZE, POOL_SUM,
        POINTWISE2_REQUANT_SHIFT, POINTWISE2_CLAMP_LOW, POINTWISE2_CLAMP_HIGH,
        POINTWISE2_WEIGHT_OFFSET, POINTWISE2_BIAS_OFFSET, SECOND_HALF, FIRST_HALF, STORE_ACTIVATIONS
    },
    {   // HIDDEN fully connected layer (input channel = reduction index / 16, position = reduction index % 16)
        MODE_POINTWISE, 1, 1, HIDDEN_CHANNEL_GROUPS, 1, FEATURE_VECTOR_LENGTH, 1, 4, 0,
        FEATURE_MAP_SIZE, 1, POOL_NONE,
        HIDDEN_REQUANT_SHIFT, HIDDEN_CLAMP_LOW, HIDDEN_CLAMP_HIGH,
        HIDDEN_WEIGHT_OFFSET, HIDDEN_BIAS_OFFSET, FIRST_HALF, SECOND_HALF, STORE_ACTIVATIONS
    },
    {   // OUTPUT fully connected layer: the scores
        MODE_POINTWISE, 1, 1, OUTPUT_CHANNEL_GROUPS, 1, HIDDEN_NEURONS, 1, 0, 0,
        1, 1, POOL_NONE, 1, 0, 0,
        OUTPUT_WEIGHT_OFFSET, OUTPUT_BIAS_OFFSET, SECOND_HALF, FIRST_HALF, STORE_SCORES
    },
};

// ============================================================
// IMAGE LOAD
// ============================================================
// image_gray8 holds the 4900 raw bytes, little-endian packed in 1225 words.
static void load_image(const uint32_t image_gray8[IMAGE_WORDS], ap_uint<32> image_words[IMAGE_WORDS])
{
#pragma HLS INLINE off

    for (int word_index = 0; word_index < IMAGE_WORDS; word_index++) {
#pragma HLS PIPELINE II=1
        image_words[word_index] = image_gray8[word_index];
    }
}

// ============================================================
// RUN_LAYER (runs STEM, DEPTHWISE, POINTWISE and fully connected layers)
// ============================================================
// One pipelined loop over (output position, channel group, pool position, reduction index).
// Counters are updated incrementally, so the loop has no multipliers or dividers besides the lane multipliers.
static void run_layer(
    const LayerConfig layer,
    const ap_uint<32> image_words[IMAGE_WORDS],
    data_t feature_buffer[LANES][2 * BUFFER_WORDS_PER_BANK],
    accumulator_t score_accumulators[OUTPUT_CHANNEL_GROUPS * LANES])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=WEIGHT_ROM complete dim=1
#pragma HLS ARRAY_PARTITION variable=BIAS_ROM complete dim=1
#pragma HLS BIND_STORAGE variable=WEIGHT_ROM type=rom_1p impl=lutram    // saves one BRAM18 per lane
    // the source and destination halves of the feature buffer never overlap, so there is no loop-carried dependence
#pragma HLS DEPENDENCE variable=feature_buffer inter false

    accumulator_t accumulator[LANES];             // running sum of every lane
    data_t pooled_value[LANES];                   // running max / sum over the pool positions

    const int total_iterations = layer.output_positions * layer.channel_groups * layer.pool_positions * layer.reduction_length;
    const int sub_index_mask = (1 << layer.input_channel_shift) - 1;

    // -------- COUNTERS (all start at the beginning of the layer) --------
    int reduction_index = 0;                      // which multiplication of the reduction
    int tap_column = 0, tap_row = 0;              // position inside the stem / depthwise window
    int sub_index = 0;                            // fully connected: position inside the input channel's 4x4 map
    int input_bank = 0;                           // pointwise: bank of the current input channel
    int pool_position = 0;                        // which of the pooled positions
    int channel_group = 0;                        // which group of LANES output channels
    int output_x = 0, output_y = 0;               // output position
    int input_channel_offset = 0;                 // pointwise: word offset of the current input channel's group
    int input_group_offset = 0;                   // depthwise: word offset of the current channel group (input buffer)
    int output_group_offset = 0;                  // word offset of the current channel group (output buffer)
    int output_row_offset = 0;                    // output_y * output_row_width
    int input_row_offset = 0;                     // 2 * output_y * input_row_width (pooling reads two input rows per output row)
    int weight_address = layer.weight_offset;     // weights are read in order, one word per cycle and lane

    for (int iteration = 0; iteration < total_iterations; iteration++) {
#pragma HLS PIPELINE II=1

        const bool last_reduction = reduction_index == layer.reduction_length - 1;
        const bool last_pool_position = pool_position == layer.pool_positions - 1;
        const bool last_channel_group = channel_group == layer.channel_groups - 1;
        const int pool_row = pool_position >> 1, pool_column = pool_position & 1;

        // -------- INPUT FETCH (the same word address in every bank) --------
        int buffer_address = 0;
        int bank_select = 0;
        bool tap_inside_map = true;
        if (layer.addressing_mode == MODE_DEPTHWISE) {
            const int input_y = output_y + tap_row - 1, input_x = output_x + tap_column - 1;
            tap_inside_map = input_y >= 0 && input_y < layer.output_row_width && input_x >= 0 && input_x < layer.output_row_width;
            // padded taps still cost a cycle, they multiply 0
            buffer_address = tap_inside_map
                ? input_group_offset + output_row_offset + (tap_row == 0 ? -layer.output_row_width : (tap_row == 1 ? 0 : layer.output_row_width))
                      + output_x + tap_column - 1
                : 0;
        } else if (layer.addressing_mode == MODE_POINTWISE) {
            buffer_address = input_channel_offset + input_row_offset + (pool_row ? layer.input_row_width : 0)
                           + 2 * output_x + pool_column + sub_index;
            bank_select = input_bank;
        }
        data_t bank_value[LANES];                 // what every bank delivers this cycle
        for (int lane = 0; lane < LANES; lane++) {
#pragma HLS UNROLL
            bank_value[lane] = feature_buffer[lane][layer.source_offset + buffer_address];
        }

        // -------- BROADCAST INPUT (stem pixel, or the current input channel for pointwise / fully connected) --------
        // stem: pixel of the stride-4 patch, ink = 255 - pixel (padding counts as no ink)
        const int image_y = 4 * output_y + tap_row - STEM_PADDING, image_x = 4 * output_x + tap_column - STEM_PADDING;
        const bool pixel_inside_image = image_y >= 0 && image_y < IMAGE_HEIGHT && image_x >= 0 && image_x < IMAGE_WIDTH;
        const int pixel_index = (layer.addressing_mode == MODE_STEM && pixel_inside_image) ? image_y * IMAGE_WIDTH + image_x : 0;
        const ap_uint<32> image_word = image_words[pixel_index >> 2];
        const ap_uint<8> pixel = image_word >> (8 * (pixel_index & 3));
        const data_t ink = pixel_inside_image ? data_t(255 - pixel) : data_t(0);
        const data_t broadcast_input = layer.addressing_mode == MODE_STEM ? ink : bank_value[bank_select];

        // -------- ONE MULTIPLY AND ADD PER LANE, REQUANTISE AND POOL at the end of the reduction --------
        for (int lane = 0; lane < LANES; lane++) {
#pragma HLS UNROLL
            const data_t lane_input = layer.addressing_mode == MODE_DEPTHWISE
                ? (tap_inside_map ? bank_value[lane] : data_t(0))
                : broadcast_input;
            const weight_t weight = WEIGHT_ROM[lane][weight_address];
            const accumulator_t accumulator_start = reduction_index == 0 ? BIAS_ROM[lane][layer.bias_offset + channel_group]
                                                                          : accumulator[lane];
            const accumulator_t new_sum = accumulator_start + weight * lane_input;
            accumulator[lane] = new_sum;

            if (last_reduction) {
                if (layer.store_mode == STORE_SCORES) {
                    score_accumulators[channel_group * LANES + lane] = new_sum;
                } else {
                    // round-half-up shift, clamp (the lower clamp of 0 is the ReLU)
                    ap_int<accumulator_t::width + 1> rounded = new_sum;
                    rounded += (ap_int<accumulator_t::width + 1>)1 << (layer.requant_shift - 1);
                    rounded >>= layer.requant_shift;
                    const data_t activation = rounded < layer.clamp_low ? data_t(layer.clamp_low)
                                            : (rounded > layer.clamp_high ? data_t(layer.clamp_high) : data_t(rounded));

                    // pooling over the 2x2 positions: sum (the /4 of the mean is folded into the next layer) or maximum;
                    // a layer without pooling has one position, where the maximum is the activation itself
                    const data_t pooled = layer.pooling == POOL_SUM ? data_t((pool_position == 0 ? data_t(0) : pooled_value[lane]) + activation)
                                        : ((pool_position == 0 || activation > pooled_value[lane]) ? activation : pooled_value[lane]);
                    pooled_value[lane] = pooled;

                    if (last_pool_position)
                        feature_buffer[lane][layer.destination_offset + output_group_offset + output_row_offset + output_x] = pooled;
                }
            }
        }

        // -------- ADVANCE THE COUNTERS --------
        if (last_reduction) {
            reduction_index = 0; tap_column = 0; tap_row = 0; sub_index = 0; input_bank = 0; input_channel_offset = 0;
            if (last_pool_position) {
                pool_position = 0;
                if (last_channel_group) {
                    channel_group = 0; input_group_offset = 0; output_group_offset = 0; weight_address = layer.weight_offset;
                    if (output_x == layer.output_row_width - 1) {
                        output_x = 0; output_y++;
                        output_row_offset += layer.output_row_width;
                        input_row_offset += 2 * layer.input_row_width;
                    } else {
                        output_x++;
                    }
                } else {
                    channel_group++;
                    input_group_offset += layer.input_group_stride;
                    output_group_offset += layer.output_group_stride;
                    weight_address++;
                }
            } else {
                pool_position++;
                weight_address -= layer.reduction_length - 1;     // the next pool position reuses the same weights
            }
        } else {
            reduction_index++; weight_address++;
            if (tap_column == layer.taps_per_row - 1) { tap_column = 0; tap_row++; } else tap_column++;
            if (sub_index == sub_index_mask) {
                sub_index = 0;
                if (input_bank == LANES - 1) { input_bank = 0; input_channel_offset += layer.input_group_stride; }
                else input_bank++;
            } else {
                sub_index++;
            }
        }
    }
}

// ============================================================
// TOP
// ============================================================
// Confidence gate: when the best score beats the second best by this margin (13.1 in Q16.16 score units, chosen on validation only)
// the selector's own top-1 class is as good as the cascade, so the controller skips the head.
static const int32_t CONFIDENT_MARGIN_Q16 = 858522;

// Interface for the cascade controller (dfx/hdl/cascade_controller.vhd), which runs the network without the PS:
//   ap_ctrl_hs (the controller drives ap_start, waits for ap_done), m_axi at fixed addresses: the image is read, the scores are written,
//   feature_map = direct write port to the RAM inside the controller, which keeps the map while the head slot is reloaded,
//   ap_return: bits 1:0 = class with the LOWEST score (the excluded one, picks the head), bits 3:2 = class with the HIGHEST score,
//   bit 4 = confident. The first class wins a tie.
extern "C" uint8_t shape_accel(
    const uint32_t image_gray8[IMAGE_WORDS],
    int32_t scores_q16[OUTPUT_COUNT],
    int16_t feature_map[PASSED_FEATURE_COUNT])
{
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE m_axi port=image_gray8 depth=1225 offset=direct bundle=gmem
#pragma HLS INTERFACE m_axi port=scores_q16 depth=OUTPUT_COUNT offset=direct bundle=gmem
#pragma HLS INTERFACE bram port=feature_map storage_type=ram_1p

    ap_uint<32> image_words[IMAGE_WORDS];
    data_t feature_buffer[LANES][2 * BUFFER_WORDS_PER_BANK];   // ping-pong feature maps in one bank: layer outputs alternate halves
    accumulator_t score_accumulators[OUTPUT_CHANNEL_GROUPS * LANES];
#pragma HLS ARRAY_PARTITION variable=feature_buffer complete dim=1

    // -------- IMAGE --------
    load_image(image_gray8, image_words);

    // -------- STEM, DEPTHWISE / POINTWISE STAGES, HIDDEN AND OUTPUT LAYERS (same engine each time) --------
    for (int layer_index = 0; layer_index < LAYER_COUNT; layer_index++) {
        run_layer(LAYER_TABLE[layer_index], image_words, feature_buffer, score_accumulators);
        if (layer_index == POINTWISE1_LAYER) {      // POINTWISE 1 + max-pool just wrote the 8x8 map into the first buffer half
            for (int channel = 0; channel < POINTWISE1_CHANNELS; channel++)
                for (int position = 0; position < POOLED_MAP_SIZE; position++) {
#pragma HLS PIPELINE II=1
                    feature_map[channel * POOLED_MAP_SIZE + position]
                        = (int16_t)feature_buffer[channel % LANES][FIRST_HALF + (channel / LANES) * POOLED_MAP_SIZE + position];
                }
        }
    }

    // -------- SCORES: the score scale is 2^-(16 - SCORE_SHIFT), so Q16.16 is an exact left shift --------
    int32_t score_values[OUTPUT_COUNT];
    for (int output = 0; output < OUTPUT_COUNT; output++) {
        score_values[output] = (int32_t)(ap_int<32>(score_accumulators[output]) << SCORE_SHIFT);
        scores_q16[output] = score_values[output];
    }
    int lowest = 0, highest = 0;
    for (int output = 1; output < OUTPUT_COUNT; output++) {
        if (score_values[output] < score_values[lowest]) lowest = output;       // the first class wins a tie
        if (score_values[output] > score_values[highest]) highest = output;
    }
    int32_t second_highest = INT32_MIN;
    for (int output = 0; output < OUTPUT_COUNT; output++)
        if (output != highest && score_values[output] > second_highest) second_highest = score_values[output];
    const bool confident = (int64_t)score_values[highest] - (int64_t)second_highest >= CONFIDENT_MARGIN_Q16;
    return (uint8_t)(lowest | (highest << 2) | ((confident ? 1 : 0) << 4));
}
