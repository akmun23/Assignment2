// nn.cpp - standalone fully spiking SNN: raw 70x70 grayscale image -> 3 class scores; the class is the highest score.
//   image -> STEM (9x9, stride 6) -> DEPTHWISE (3x3, stride 2) -> POINTWISE -> HIDDEN -> OUTPUT
// Every layer runs LIF for six steps: membrane = (membrane >> decay_shift) + current; fire at threshold, then subtract it once.
// The stem supplies constant current from ink = 255 - pixel; later layers exchange spikes. Each weight read updates all six
// time-step currents in parallel across six lanes; only the local membrane recurrence runs in time. Readout combines the
// output spike count times threshold with the residual membrane, then floors to Q16.16 scores. All arithmetic is integer.
#include "ap_int.h"
#include <stdint.h>

// ============================================================
// NETWORK CONFIGURATION (derived from the trained model; the weight tables are in nn_weights.h)
// ============================================================
// Source model w_p6h96; signed weights /64, stem bias /16384, later biases /64.
// Input-independent intervals include every reduction prefix and pre-reset state.
typedef ap_int<8> weight_t;               // signed synaptic weights
typedef ap_int<13> current_t;             // current interval [-2411, 2427]
typedef ap_int<14> membrane_t;            // membrane interval [-5436, 4716]
typedef ap_int<20> analog_accumulator_t;  // stem sum including rounding [-294158, 157581]
typedef ap_uint<6> spike_train_t;         // one neuron's six time steps
typedef ap_uint<36> packed_spikes_t;      // six lane-local spike trains

// Network sizes and temporal configuration.
static const int LANES = 6;
static const int IMAGE_HEIGHT = 70;
static const int IMAGE_WIDTH = 70;
static const int IMAGE_WORDS = IMAGE_HEIGHT * IMAGE_WIDTH / 4;
static const int OUTPUT_COUNT = 3;
static const int LIF_STEPS = 6;
static const int LAYER_COUNT = 5;
static const int KERNEL_COUNTER_BITS = 4;
static const int BUFFER_WORDS_PER_BANK = 242;
static const int STEM_CURRENT_SHIFT = 8;
static const int STEM_ROUNDING_OFFSET = 128;
static const int OUTPUT_THRESHOLD_EXPONENT = 6;
static const int READOUT_MODE = 0;         // spike count times threshold plus residual membrane
static const int SCORE_DENOMINATOR = 384;

// ROMs are banked by output lane, then [channel group][reduction index].
// Layer order: stem, depthwise, pointwise, hidden, output. Unused lanes are zero.
static const int WEIGHT_ROM_SIZE = 3744;
static const int BIAS_ROM_SIZE = 22;

// ============================================================
// LAYER TABLE
// ============================================================
enum AddressingMode {
    MODE_STEM = 0,          // raw image patches, shared by all output lanes
    MODE_CONVOLUTION = 1,   // input channels and spatial taps, shared by all lanes
    MODE_DEPTHWISE = 2,     // each lane reads its own channel's spatial taps
    MODE_CONNECTED = 3     // channel-major flattened input, shared by all lanes
};

struct LayerConfiguration {
    int addressing_mode;
    int input_channels;
    int input_width;
    int input_positions;    // spatial positions per input channel
    int output_channels;
    int output_width;
    int output_positions;   // spatial positions per output channel
    int channel_groups;     // groups of LANES output channels
    int kernel_size;        // input map width for a fully connected layer
    int stride;
    int padding;
    int pool_size;          // must be one; the engine has no pooling loops
    int reduction_length;   // weights accumulated into one output neuron
    int threshold_exponent;
    int decay_shift;
    int weight_offset;      // layer start in each lane's weight ROM
    int bias_offset;        // layer start in each lane's bias ROM
};

// Fields: mode, input channels/width/positions, output channels/width/positions,
// groups, kernel, stride, padding, pool, reduction, threshold, decay, weight, bias.
static constexpr LayerConfiguration LAYER_TABLE[LAYER_COUNT] = {
    {0, 1, 70, 4900, 12, 11, 121, 2, 9, 6, 1, 1, 81, 6, 1, 0, 0}, // STEM: 9x9 stride 6
    {2, 12, 11, 121, 12, 6, 36, 2, 3, 2, 1, 1, 9, 6, 1, 162, 2}, // DEPTHWISE: 3x3 stride 2
    {1, 12, 6, 36, 6, 6, 36, 1, 1, 1, 0, 1, 12, 6, 1, 180, 4}, // POINTWISE: 1x1
    {3, 6, 6, 36, 96, 1, 1, 16, 6, 0, 0, 1, 216, 6, 1, 192, 5}, // HIDDEN: fully connected
    {3, 96, 1, 1, 3, 1, 1, 1, 1, 0, 0, 1, 96, 6, 0, 3648, 21}, // OUTPUT: fully connected
};

// Constant shift/add addressing avoids general dividers and address DSPs.
static int row_offset(int index, int width)
{
#pragma HLS INLINE
    switch (width) {
    case 1: return int(uint32_t(index));
    case 6: return int((uint32_t(index) << 1) + (uint32_t(index) << 2));
    case 11: return int(uint32_t(index) + (uint32_t(index) << 1) + (uint32_t(index) << 3));
    case 70: return int((uint32_t(index) << 1) + (uint32_t(index) << 2) + (uint32_t(index) << 6));
    default: return 0; // Only generated layer-table values occur.
    }
}

static int group_offset(int index, int positions)
{
#pragma HLS INLINE
    switch (positions) {
    case 1: return int(uint32_t(index));
    case 36: return int((uint32_t(index) << 2) + (uint32_t(index) << 5));
    case 121: return int(uint32_t(index) + (uint32_t(index) << 3) + (uint32_t(index) << 4) + (uint32_t(index) << 5) + (uint32_t(index) << 6));
    case 4900: return int((uint32_t(index) << 2) + (uint32_t(index) << 5) + (uint32_t(index) << 8) + (uint32_t(index) << 9) + (uint32_t(index) << 12));
    default: return 0; // Only generated layer-table values occur.
    }
}

static int stride_product(int index, int stride)
{
#pragma HLS INLINE
    switch (stride) {
    case 0: return int(0);
    case 1: return int(uint32_t(index));
    case 2: return int((uint32_t(index) << 1));
    case 6: return int((uint32_t(index) << 1) + (uint32_t(index) << 2));
    default: return 0; // Only generated layer-table values occur.
    }
}

static int weight_group_offset(int index, int reduction)
{
#pragma HLS INLINE
    switch (reduction) {
    case 9: return int(uint32_t(index) + (uint32_t(index) << 3));
    case 12: return int((uint32_t(index) << 2) + (uint32_t(index) << 3));
    case 81: return int(uint32_t(index) + (uint32_t(index) << 4) + (uint32_t(index) << 6));
    case 96: return int((uint32_t(index) << 5) + (uint32_t(index) << 6));
    case 216: return int((uint32_t(index) << 3) + (uint32_t(index) << 4) + (uint32_t(index) << 6) + (uint32_t(index) << 7));
    default: return 0; // Only generated layer-table values occur.
    }
}

// ============================================================
// END NETWORK CONFIGURATION
// ============================================================

#include "nn_weights.h"   // only the weight and bias tables

// ============================================================
// CONFIGURATION CHECKS
// ============================================================
static_assert(LANES * LIF_STEPS == packed_spikes_t::width,
              "Packed feature word width must match lanes and steps");
static_assert(LAYER_TABLE[0].pool_size == 1 && LAYER_TABLE[1].pool_size == 1 &&
              LAYER_TABLE[2].pool_size == 1 && LAYER_TABLE[3].pool_size == 1 &&
              LAYER_TABLE[4].pool_size == 1, "Final engine requires unpooled layers");
static_assert(WEIGHT_ROM_SIZE == sizeof(WEIGHT_ROM[0]) / sizeof(WEIGHT_ROM[0][0]) &&
              BIAS_ROM_SIZE == sizeof(BIAS_ROM[0]) / sizeof(BIAS_ROM[0][0]),
              "Generated ROM dimensions must match the configuration");
static_assert(LANES == 6 && LIF_STEPS == 6 && LAYER_COUNT == 5,
              "Final engine requires six lanes, six steps and five layers");
static_assert(current_t::width == 13 && membrane_t::width == 14 && analog_accumulator_t::width == 20,
              "Rerun the input-independent bounds proof before changing widths");

// ============================================================
// SCORE QUANTISATION
// ============================================================
static int32_t output_score(int32_t readout_units)
{
#pragma HLS INLINE off
    ap_int<48> scaled_units = readout_units;
    scaled_units <<= 18;
    if (LIF_STEPS != 6) {
        scaled_units >>= OUTPUT_THRESHOLD_EXPONENT + (LIF_STEPS == 4 ? 2 : 3);
        return int32_t(scaled_units);
    }
    // One shared, serial divider is used only for the three final scores. Export bounds
    // prove |scaled_units| < 3*2^31; a wide native /3 would infer many DSP multipliers.
    scaled_units >>= OUTPUT_THRESHOLD_EXPONENT + 1;
    const bool negative = scaled_units < 0;
    const ap_uint<34> magnitude = negative ? ap_uint<34>(-scaled_units) : ap_uint<34>(scaled_units);
    ap_uint<34> quotient = 0;
    ap_uint<2> remainder = 0;
    for (int bit = 33; bit >= 0; bit--) {
#pragma HLS PIPELINE II=1
        const ap_uint<3> trial = (ap_uint<3>(remainder) << 1) | ap_uint<3>(magnitude[bit]);
        const bool subtract = trial >= 3;
        remainder = subtract ? ap_uint<2>(trial - 3) : ap_uint<2>(trial);
        quotient[bit] = subtract;
    }
    ap_int<35> signed_score = quotient;
    if (negative) {
        signed_score = -signed_score;
        if (remainder != 0)
            signed_score--;
    }
    return int32_t(signed_score);
}

// ============================================================
// RUN_LAYER (stem, depthwise, pointwise and fully connected layers)
// ============================================================
static void run_layer(const LayerConfiguration &layer, int source_offset, int destination_offset,
                      const ap_uint<32> image_words[IMAGE_WORDS],
                      packed_spikes_t feature_buffer[2 * BUFFER_WORDS_PER_BANK],
                      int32_t final_units[OUTPUT_COUNT])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=WEIGHT_ROM complete dim=1
#pragma HLS ARRAY_PARTITION variable=BIAS_ROM complete dim=1
#pragma HLS BIND_STORAGE variable=WEIGHT_ROM type=rom_1p impl=lutram    // weights in logic, not block RAM: BRAM contents would add to the partial bitstream
    // Each output position and channel group reuses the same lane hardware.
    for (int output_y = 0; output_y < layer.output_width; output_y++) {
        for (int output_x = 0; output_x < layer.output_width; output_x++) {
            for (int group = 0; group < layer.channel_groups; group++) {
                // -------- INITIAL CURRENTS AND STEM SUMS --------
                current_t currents[LANES][LIF_STEPS];
                analog_accumulator_t analog_sums[LANES];
#pragma HLS ARRAY_PARTITION variable=currents complete dim=0
#pragma HLS ARRAY_PARTITION variable=analog_sums complete
                for (int lane = 0; lane < LANES; lane++) {
#pragma HLS UNROLL
                    const ap_int<32> bias = BIAS_ROM[lane][layer.bias_offset + group];
                    analog_sums[lane] = bias;
                    for (int step = 0; step < LIF_STEPS; step++) {
#pragma HLS UNROLL
                        currents[lane][step] = layer.addressing_mode == MODE_STEM ? current_t(0) : current_t(bias);
                    }
                }
                // -------- INPUT ADDRESSES AND TIME-PARALLEL REDUCTION --------
                ap_uint<3> input_lane = 0;
                ap_uint<5> input_group = 0;
                ap_uint<KERNEL_COUNTER_BITS> kernel_y = 0, kernel_x = 0;
                const int origin_y = stride_product(output_y, layer.stride) - layer.padding;
                const int origin_x = stride_product(output_x, layer.stride) - layer.padding;
                const int weight_base = layer.weight_offset + weight_group_offset(group, layer.reduction_length);
                for (int reduction = 0; reduction < layer.reduction_length; reduction++) {
#pragma HLS PIPELINE II=1
                    const int input_y = origin_y + int(kernel_y);
                    const int input_x = origin_x + int(kernel_x);
                    const bool inside = input_y >= 0 && input_y < layer.input_width && input_x >= 0 && input_x < layer.input_width;
                    const int position = inside ? row_offset(input_y, layer.input_width) + input_x : 0;
                    const int shared_address = source_offset + group_offset(input_group, layer.input_positions) + position;
                    const int depthwise_address = source_offset + group_offset(group, layer.input_positions) + position;
                    // Every lane at one feature address fits a single BRAM word.
                    const int feature_address = layer.addressing_mode == MODE_DEPTHWISE
                        ? depthwise_address : shared_address;
                    const packed_spikes_t input_spike_trains = inside && layer.addressing_mode != MODE_STEM
                        ? feature_buffer[feature_address] : packed_spikes_t(0);
                    const spike_train_t shared_spike_train = input_spike_trains >> (LIF_STEPS * int(input_lane));
                    const int pixel_index = inside ? input_y * IMAGE_WIDTH + input_x : 0;
                    const ap_uint<8> pixel = layer.addressing_mode == MODE_STEM
                        ? ap_uint<8>(image_words[pixel_index >> 2] >> (8 * (pixel_index & 3))) : ap_uint<8>(255);
                    const ap_uint<8> ink = inside ? ap_uint<8>(255 - pixel) : ap_uint<8>(0);
                    for (int lane = 0; lane < LANES; lane++) {
#pragma HLS UNROLL
                        const weight_t weight = WEIGHT_ROM[lane][weight_base + reduction];
                        if (layer.addressing_mode == MODE_STEM) {
                            const ap_int<18> product = weight * ap_int<9>(ink);
#pragma HLS BIND_OP variable=product op=mul impl=dsp
                            analog_sums[lane] += product;
                        } else {
                            const spike_train_t spike_train = layer.addressing_mode == MODE_DEPTHWISE
                                ? spike_train_t(input_spike_trains.range((lane + 1) * LIF_STEPS - 1, lane * LIF_STEPS)) : shared_spike_train;
                            // A zero word disables every add without event-list construction.
                            for (int step = 0; step < LIF_STEPS; step++) {
#pragma HLS UNROLL
                                if (inside && spike_train[step])
                                    currents[lane][step] += weight;
                            }
                        }
                    }
                    kernel_x++;
                    if (kernel_x == layer.kernel_size) {
                        kernel_x = 0;
                        kernel_y++;
                        if (kernel_y == layer.kernel_size) {
                            kernel_y = 0;
                            input_lane++;
                            if (input_lane == LANES) {
                                input_lane = 0;
                                input_group++;
                            }
                        }
                    }
                }
                // -------- STEM ROUNDING (no pooling; other currents are already complete) --------
                if (layer.addressing_mode == MODE_STEM) {
                    for (int lane = 0; lane < LANES; lane++) {
#pragma HLS UNROLL
                        const current_t constant_current = (analog_sums[lane] + STEM_ROUNDING_OFFSET) >> STEM_CURRENT_SHIFT;
                        for (int step = 0; step < LIF_STEPS; step++) {
#pragma HLS UNROLL
                            currents[lane][step] = constant_current;
                        }
                    }
                }
                // -------- ORDERED LIF RECURRENCE --------
                membrane_t membranes[LANES];
                spike_train_t spike_trains[LANES];
                ap_uint<3> spike_counts[LANES];
#pragma HLS ARRAY_PARTITION variable=membranes complete
#pragma HLS ARRAY_PARTITION variable=spike_trains complete
#pragma HLS ARRAY_PARTITION variable=spike_counts complete
                for (int lane = 0; lane < LANES; lane++) {
#pragma HLS UNROLL
                    membranes[lane] = 0;
                    spike_trains[lane] = 0;
                    spike_counts[lane] = 0;
                }
                for (int step = 0; step < LIF_STEPS; step++) {
#pragma HLS PIPELINE II=2
                    for (int lane = 0; lane < LANES; lane++) {
#pragma HLS UNROLL
                        const membrane_t current = currents[lane][step];
                        membrane_t membrane = (membranes[lane] >> layer.decay_shift) + current;
                        const int threshold = 1 << layer.threshold_exponent;
                        const bool fired = membrane >= threshold;
                        if (fired)
                            membrane -= threshold;
                        membranes[lane] = membrane;
                        spike_trains[lane][step] = fired;
                        spike_counts[lane] += fired ? 1 : 0;
                    }
                }
                // -------- PACK SPIKES AND CAPTURE OUTPUT READOUT --------
                const int destination = destination_offset + group_offset(group, layer.output_positions)
                                        + row_offset(output_y, layer.output_width) + output_x;
                packed_spikes_t output_trains = 0;
                for (int lane = 0; lane < LANES; lane++) {
#pragma HLS UNROLL
                    output_trains.range((lane + 1) * LIF_STEPS - 1, lane * LIF_STEPS) = spike_trains[lane];
                    if (layer.addressing_mode == MODE_CONNECTED && layer.output_channels == OUTPUT_COUNT && lane < OUTPUT_COUNT) {
                        const int32_t readout_units = (READOUT_MODE == 1 ? 0 : int32_t(spike_counts[lane]) * (1 << OUTPUT_THRESHOLD_EXPONENT))
                                              + (READOUT_MODE == 2 ? 0 : int32_t(membranes[lane]));
                        final_units[lane] = readout_units;
                    }
                }
                feature_buffer[destination] = output_trains;
            }
        }
    }
}

// ============================================================
// TOP
// ============================================================
// Interface for the cascade controller (dfx/hdl/cascade_controller.vhd), which runs the network without the PS:
//   ap_ctrl_hs (the controller drives ap_start, waits for ap_done), ap_return = the class with the HIGHEST score (the first wins a tie),
//   m_axi at fixed addresses (offset ports tied to constants): the image is read, the scores are written.
extern "C" uint8_t shape_accel(const uint32_t image_gray8[IMAGE_WORDS], int32_t scores_q16[OUTPUT_COUNT])
{
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE m_axi port=image_gray8 depth=IMAGE_WORDS offset=direct bundle=gmem max_read_burst_length=16 max_write_burst_length=2 num_read_outstanding=1 num_write_outstanding=1
#pragma HLS INTERFACE m_axi port=scores_q16 depth=OUTPUT_COUNT offset=direct bundle=gmem max_read_burst_length=16 max_write_burst_length=2 num_read_outstanding=1 num_write_outstanding=1
    ap_uint<32> image_words[IMAGE_WORDS];
    packed_spikes_t feature_buffer[2 * BUFFER_WORDS_PER_BANK];
    int32_t final_units[OUTPUT_COUNT];
#pragma HLS ARRAY_PARTITION variable=final_units complete
#pragma HLS BIND_STORAGE variable=feature_buffer type=ram_s2p impl=bram
    // -------- IMAGE (4900 bytes, little-endian packed four per word) --------
    for (int word = 0; word < IMAGE_WORDS; word++) {
#pragma HLS PIPELINE II=1
        image_words[word] = image_gray8[word];
    }
    // -------- ALL LAYERS (alternate halves of the packed spike buffer) --------
    int source_offset = BUFFER_WORDS_PER_BANK;
    int destination_offset = 0;
    for (int layer = 0; layer < LAYER_COUNT; layer++) {
        run_layer(LAYER_TABLE[layer], source_offset, destination_offset, image_words, feature_buffer, final_units);
        const int previous_source = source_offset;
        source_offset = destination_offset;
        destination_offset = previous_source;
    }
    // -------- SCORES AND CLASS (strict comparison preserves first-wins ties) --------
    uint8_t class_result = 0;
    int32_t chosen_score = 0;
    for (int output = 0; output < OUTPUT_COUNT; output++) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=1
        const int32_t score = output_score(final_units[output]);
        scores_q16[output] = score;
        if (output == 0 || score > chosen_score) {
            chosen_score = score;
            class_result = uint8_t(output);
        }
    }
    return class_result;
}
