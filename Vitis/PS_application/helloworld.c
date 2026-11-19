// helloworld.c - PS application of Assignment 2 (adapted from CNN_lecture/code/helloworld.c)
//
// Receives a 70x70 grayscale image from the PC over UART (pc/protocol.py), puts it into the BRAM and lets the cascade controller
// (dfx/hdl/cascade_controller.vhd) run the three paths: "cnn", "snn" and "cascade" (NN_selector, then ONE head for the two classes it kept, unless the selector is confident).
// The controller loads the networks through the ICAP and runs them on its own; after the one-time DFX setup the PS only starts it, waits,
// and reads the result registers and the scores the networks wrote into the BRAM. One reply record per path.
//
// UART protocol (pc/protocol.py, little-endian):
//   PC -> PS   0xA6 0x6A <length u16 = 4900> <4900 image bytes, row-major, 255 = white>
//   PS -> PC   0xA5 <uart_rx_us u32> 3 x { <pred u8> <flags u8> <net_load_ns u32> <net_run_ns u32> <head_load_ns u32> <head_run_ns u32>
//              <total_ns u32> <score0 s16> <score1 s16> <score2 s16> }  0x5A
//              flags: bit 0 = a head ran, bits 2:1 = the head (the class the selector excluded); uart_rx_us = time to receive the image payload
#include <stdio.h>
#include <stdint.h>

#include "platform.h"
#include "xil_printf.h"
#include "xil_io.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xiltimer.h"
#include "sleep.h"
#include "xuartps.h"

#include "xdevcfg.h"   // device configuration controller: hands partial reconfiguration over from the PCAP to the ICAP
#include "xprc.h"      // DFX Controller driver (Vitis library "prc")

// Partial bitstreams, linked into the ELF so that they are in DDR as soon as the application runs
// (byte-swapped and converted with xxd -i).
#include "partial_rm_cnn.h"
#include "partial_rm_snn.h"
#include "partial_rm_selector.h"
#include "partial_rm_head_circle.h"
#include "partial_rm_head_square.h"
#include "partial_rm_head_triangle.h"

extern char inbyte(void);
extern void outbyte(char c);

// ============================================================
// UART BAUD CONFIG
// ============================================================
#define UART_BASE_ADDR   XPAR_XUARTPS_0_BASEADDR       // the SDT platform looks the driver up by base address
#define UART_BAUD        4000000u                      // pc/draw.py default

// ============================================================
// Input image (70x70) and frame protocol
// ============================================================
#define IMG_H     70
#define IMG_W     70
#define N_INPUTS  (IMG_H * IMG_W)          // 4900 bytes = 1225 32-bit words
#define RX_BYTES  N_INPUTS

#define FRAME_START_0  0xA6
#define FRAME_START_1  0x6A

#define N_CLASSES   3                      // circle, square, triangle (training label order)
#define PRED_FAILED 0xFF                   // pc/protocol.py reports "FPGA <path> failed"

// ============================================================
// BLOCK DESIGN ADDRESSES (xparameters.h, cell names of the Vivado block design)
// ============================================================
// Image memory: the AXI BRAM controller the PS fills and the networks read.
#define BRAM_BASE_ADDR  XPAR_AXI_BRAM_CTRL_0_BASEADDR
#define BRAM_BYTES      0x2000u            // 8 KB

// The cascade controller is an HDL block without a driver, so it has no name in xparameters.h; this is its Address Editor value.
#define CASCADE_BASE_ADDR  0x40030000u

// ============================================================
// LAYOUT OF THE IMAGE MEMORY (BRAM); the networks use these fixed addresses (the block design)
// ============================================================
#define IMAGE_OFFSET     0x0000u           // 4900 bytes, written by the PS (image_gray8)
#define SCORES_OFFSET    0x1400u           // 3 x int32 scores of the CNN, SNN or selector (Q16.16); the head's score follows at +12
#if (SCORES_OFFSET + 4 * N_CLASSES) > BRAM_BYTES
#error "image memory layout does not fit into the BRAM"
#endif
#define IMAGE_ADDR       (BRAM_BASE_ADDR + IMAGE_OFFSET)
#define SCORES_ADDR      (BRAM_BASE_ADDR + SCORES_OFFSET)

// ============================================================
// CASCADE CONTROLLER REGISTERS (documented in dfx/hdl/cascade_controller.vhd)
// ============================================================
#define REG_CONTROL           0x00         // write: start (bit 0) and mode (bits 2:1)
#define REG_STATUS            0x04         // busy (bit 0), done (bit 1), error (bit 2)
#define REG_RESULT            0x08         // final class (bits 1:0), excluded class (bits 9:8), head ran (bit 17)
#define REG_NET_LOAD_CYCLES   0x0C         // clock cycles of the last run: network slot loading
#define REG_NET_RUN_CYCLES    0x10         //                               network slot running
#define REG_HEAD_LOAD_CYCLES  0x14         //                               head slot loading
#define REG_HEAD_RUN_CYCLES   0x18         //                               head slot running
#define REG_TOTAL_CYCLES      0x1C         //                               from start to done
#define RESULT_EXCLUDED_SHIFT 8
#define RESULT_HEAD_RAN_BIT   17
#define CONTROL_START_BIT     0x01u
#define CONTROL_MODE_SHIFT    1
#define STATUS_DONE_BIT       1
#define STATUS_ERROR_BIT      2

#define PL_CLOCK_MHZ          100u         // fabric clock (FCLK0) the cycle counters run at
#define PL_NS_PER_CYCLE       (1000u / PL_CLOCK_MHZ)
#define PATH_TIMEOUT_US       200000u      // a path that has not finished after this long has failed

static inline void ip_write(uintptr_t base, uint32_t off, uint32_t val)
{
    Xil_Out32(base + off, val);
}

static inline uint32_t ip_read(uintptr_t base, uint32_t off)
{
    return Xil_In32(base + off);
}

// The three paths, in the order of the reply records; the value is also the mode of the cascade controller.
typedef enum { PATH_CNN = 0, PATH_SNN = 1, PATH_CASCADE = 2, N_PATHS = 3 } Path;
static const char *const path_name[N_PATHS] = {"cnn", "snn", "cascade"};
static const char *const class_name[N_CLASSES] = {"circle", "square", "triangle"};

typedef struct {
    uint8_t  pred;                         // class index, or PRED_FAILED
    uint8_t  flags;                        // bit 0: a head ran, bits 2:1: which head (the class the selector excluded)
    uint32_t net_load_ns, net_run_ns;      // network slot: time spent loading (0 when the module was already there) and running
    uint32_t head_load_ns, head_run_ns;    // head slot, same
    uint32_t total_ns;                     // start to done in the PL
    int16_t  scores[N_CLASSES];            // Q8.8, see score_q16_to_q8_8()
} PathResult;

// ============================================================
// TIME
// ============================================================
// The global timer starts at the first usleep() (see main). COUNTS_PER_SECOND is defined as "XPAR_..._FREQ_HZ/2" without
// parentheses, hence the extra pair.
static uint32_t time_us_since(XTime t0)
{
    XTime t1;
    XTime_GetTime(&t1);
    return (uint32_t)(((uint64_t)(t1 - t0) * 1000000ULL) / ((uint64_t)(COUNTS_PER_SECOND)));
}

// ============================================================
// UART helpers
// ============================================================
static void uart_set_baudrate(u32 baud)
{
    XUartPs Uart;
    XUartPs_Config *Cfg = XUartPs_LookupConfig(UART_BASE_ADDR);
    if (!Cfg) return;

    if (XUartPs_CfgInitialize(&Uart, Cfg, Cfg->BaseAddress) != XST_SUCCESS)
        return;

    XUartPs_SetOperMode(&Uart, XUARTPS_OPER_MODE_NORMAL);
    (void)XUartPs_SetBaudRate(&Uart, baud);
}

static uint8_t read_byte_blocking(void)
{
    return (uint8_t)inbyte();
}

// Reads bytes until the two start bytes follow each other (also finds them after a stray 0xA6).
static void wait_for_frame_header(void)
{
    uint8_t b0 = read_byte_blocking();
    while (1) {
        uint8_t b1 = read_byte_blocking();
        if (b0 == FRAME_START_0 && b1 == FRAME_START_1)
            return;
        b0 = b1;
    }
}

// Receives one image frame. Returns 0 on success, -1 if the announced length is not RX_BYTES.
static int uart_receive_image(uint8_t *rx_buf, uint32_t *payload_us)
{
    wait_for_frame_header();

    uint16_t len_lo = read_byte_blocking();         // separate statements: the order of two calls in one expression is unspecified
    uint16_t len_hi = read_byte_blocking();
    uint16_t len = (uint16_t)(len_lo | (len_hi << 8));
    if (len != RX_BYTES) {
        xil_printf("Bad length: %d (expected %d)\r\n", len, RX_BYTES);
        return -1;
    }

    XTime t0;
    XTime_GetTime(&t0);
    for (int i = 0; i < RX_BYTES; i++)
        rx_buf[i] = read_byte_blocking();
    *payload_us = time_us_since(t0);
    return 0;
}

static void uart_send_le(uint32_t value, int n_bytes)
{
    for (int i = 0; i < n_bytes; i++)
        outbyte((char)((value >> (8 * i)) & 0xFF));
}

// ============================================================
// Reply frame (layout in the header)
// ============================================================
static void uart_send_reply_frame(uint32_t uart_rx_us, const PathResult result[N_PATHS])
{
    outbyte((char)0xA5);
    uart_send_le(uart_rx_us, 4);
    for (int p = 0; p < N_PATHS; p++) {
        outbyte((char)result[p].pred);
        outbyte((char)result[p].flags);
        uart_send_le(result[p].net_load_ns, 4);
        uart_send_le(result[p].net_run_ns, 4);
        uart_send_le(result[p].head_load_ns, 4);
        uart_send_le(result[p].head_run_ns, 4);
        uart_send_le(result[p].total_ns, 4);
        for (int c = 0; c < N_CLASSES; c++)
            uart_send_le((uint16_t)result[p].scores[c], 2);
    }
    outbyte((char)0x5A);
}

// ============================================================
// Image memory (BRAM)
// ============================================================
// The image goes into the BRAM as 1225 little-endian 32-bit words, which is how the accelerators read it (image_gray8).
// The BRAM is reached through the AXI GP port (device memory): the writes are ordered and not cached, so the accelerator
// sees the complete image once Xil_Out32 has returned.
static uint8_t rx_buf[RX_BYTES];

static void bram_store_image(const uint8_t *image)
{
    for (int i = 0; i < N_INPUTS / 4; i++) {
        uint32_t word = (uint32_t)image[4 * i]
                      | ((uint32_t)image[4 * i + 1] << 8)
                      | ((uint32_t)image[4 * i + 2] << 16)
                      | ((uint32_t)image[4 * i + 3] << 24);
        Xil_Out32(IMAGE_ADDR + 4u * (uint32_t)i, word);
    }
}

static int32_t bram_read_score(int i)
{
    return (int32_t)Xil_In32(SCORES_ADDR + 4u * (uint32_t)i);
}

// Q16.16 -> Q8.8 (rounded, saturated to int16), the format of the reply.
static int16_t score_q16_to_q8_8(int32_t score_q16)
{
    int32_t rounded = (score_q16 + 128) >> 8;
    if (rounded > 32767) return 32767;
    if (rounded < -32768) return -32768;
    return (int16_t)rounded;
}

// ============================================================
// DFX: set up the DFX Controller once at boot (the cascade controller does all loading afterwards)
// ============================================================
// The reconfigurable design (dfx/) has two partitions:
//   network slot (virtual socket 0): CNN, SNN or selector      head slot (virtual socket 1): circle, square or triangle head
// The DFX Controller (a block of the static design) fetches the partial bitstream of the wanted network from DDR through the PS
// high-performance port and writes it to the ICAP, decoupling the slot while it does so. The configuration happens only through the ICAP,
// never through the PCAP. After this setup the cascade controller raises the hardware trigger of the module it needs.
#define DFX_BASE_ADDR       XPAR_DFX_CONTROLLER_0_BASEADDR
#define DFX_TIMEOUT_US      100000u
#define DFX_N_SLOTS         2
#define DFX_N_MODULES       6

typedef struct {
    uint16_t slot;                         // virtual socket manager of the DFX Controller
    uint16_t rm_id;                        // reconfigurable module ID inside its slot = bitstream index = hardware trigger number
    unsigned char *bitstream;              // partial bitstream embedded in the ELF
    uint32_t size_bytes;
} DFX_Module;

// The RM IDs follow the order of the modules of each slot in the Vivado DFX configuration (RM IDs of the DFX Controller).
static const DFX_Module dfx_module[DFX_N_MODULES] = {
    {0, 0, partial_rm_cnn_bin,           sizeof(partial_rm_cnn_bin)},            // CNN_standalone
    {0, 1, partial_rm_snn_bin,           sizeof(partial_rm_snn_bin)},            // SNN_standalone
    {0, 2, partial_rm_selector_bin,      sizeof(partial_rm_selector_bin)},       // NN_selector
    {1, 0, partial_rm_head_circle_bin,   sizeof(partial_rm_head_circle_bin)},    // NN_circle
    {1, 1, partial_rm_head_square_bin,   sizeof(partial_rm_head_square_bin)},    // NN_square
    {1, 2, partial_rm_head_triangle_bin, sizeof(partial_rm_head_triangle_bin)},  // NN_triangle
};

static XPrc Prc;

// Waits until slot `slot` is (want_shutdown = 1) or is not (want_shutdown = 0) in shutdown. Returns 0 on success.
static int DFX_WaitForShutdown(uint16_t slot, int want_shutdown)
{
    XTime t0;
    XTime_GetTime(&t0);
    while (XPrc_IsVsmInShutdown(&Prc, slot) != want_shutdown) {
        if (time_us_since(t0) > DFX_TIMEOUT_US) {
            xil_printf("ERROR: DFX slot %d did not %s shutdown\r\n", slot, want_shutdown ? "enter" : "leave");
            return -1;
        }
    }
    return 0;
}

// The Zynq lets either the PS (PCAP) or the fabric (ICAP) do partial reconfiguration, selected in the device configuration
// controller. It comes up with the PCAP selected; without this switch the ICAP of the DFX Controller is ignored.
static int DFX_SelectIcap(void)
{
    static XDcfg DevCfg;
    XDcfg_Config *Cfg = XDcfg_LookupConfig(XPAR_DEVCFG_BASEADDR);
    if (!Cfg) {
        xil_printf("ERROR: device configuration controller not found\r\n");
        return XST_FAILURE;
    }
    if (XDcfg_CfgInitialize(&DevCfg, Cfg, Cfg->BaseAddr) != XST_SUCCESS) {
        xil_printf("ERROR: device configuration controller initialization failed\r\n");
        return XST_FAILURE;
    }
    XDcfg_SelectIcapInterface(&DevCfg);
    return XST_SUCCESS;
}

// Tells the DFX Controller where the partial bitstream of every network is and starts both slots.
static int DFX_Init(void)
{
    if (DFX_SelectIcap() != XST_SUCCESS) return XST_FAILURE;

    XPrc_Config *Cfg = XPrc_LookupConfig(DFX_BASE_ADDR);
    if (!Cfg) {
        xil_printf("ERROR: DFX Controller configuration not found\r\n");
        return XST_FAILURE;
    }
    if (XPrc_CfgInitialize(&Prc, Cfg, Cfg->BaseAddress) != XST_SUCCESS) {
        xil_printf("ERROR: DFX Controller initialization failed\r\n");
        return XST_FAILURE;
    }

    // The tables can only be written while a slot is in shutdown.
    for (uint16_t slot = 0; slot < DFX_N_SLOTS; slot++) {
        if (!XPrc_IsVsmInShutdown(&Prc, slot)) {
            XPrc_SendShutdownCommand(&Prc, slot);
            if (DFX_WaitForShutdown(slot, 1) != 0) return XST_FAILURE;
        }
    }

    for (int m = 0; m < DFX_N_MODULES; m++) {
        const DFX_Module *Rm = &dfx_module[m];
        Xil_DCacheFlushRange((UINTPTR)Rm->bitstream, Rm->size_bytes);       // the DFX Controller reads DDR, not the cache
        XPrc_SetRmBsIndex(&Prc, Rm->slot, Rm->rm_id, Rm->rm_id);            // module -> its bitstream
        XPrc_SetRmControl(&Prc, Rm->slot, Rm->rm_id, 0, 0, 0, 0);           // no software shutdown/startup/reset handshake
        XPrc_SetBsId(&Prc, Rm->slot, Rm->rm_id, Rm->rm_id);
        XPrc_SetBsAddress(&Prc, Rm->slot, Rm->rm_id, (u32)(UINTPTR)Rm->bitstream);
        XPrc_SetBsSize(&Prc, Rm->slot, Rm->rm_id, Rm->size_bytes);
        XPrc_SetTriggerToRmMapping(&Prc, Rm->slot, Rm->rm_id, Rm->rm_id);   // hardware trigger n of a slot loads module n
    }

    // Start the slots with the first module of each one, which the full bitstream delivers (the CNN and the circle head).
    for (uint16_t slot = 0; slot < DFX_N_SLOTS; slot++) {
        XPrc_SendRestartWithStatusCommand(&Prc, slot, 1, 0);
        if (DFX_WaitForShutdown(slot, 0) != 0) return XST_FAILURE;
    }
    return XST_SUCCESS;
}

// ============================================================
// THE THREE PATHS (run by the cascade controller)
// ============================================================
// Starts the cascade controller in mode `path` and waits until it is done. The controller loads the networks it needs, runs them, and for the
// cascade takes the selector's lowest score (the class it excludes) to load the head of the remaining pair and decide between its two classes.
// With the confidence gate the head is skipped when the selector is sure. CNN and SNN: the class is the highest of the three scores. Returns 0 on success.
static int run_path(Path path, PathResult *result)
{
    ip_write(CASCADE_BASE_ADDR, REG_CONTROL, ((uint32_t)path << CONTROL_MODE_SHIFT) | CONTROL_START_BIT);

    XTime t0;
    XTime_GetTime(&t0);
    uint32_t status;
    do {
        status = ip_read(CASCADE_BASE_ADDR, REG_STATUS);
        if (time_us_since(t0) > PATH_TIMEOUT_US) {
            xil_printf("%s did not finish within %d us\r\n", path_name[path], (int)PATH_TIMEOUT_US);
            return -1;
        }
    } while (((status >> STATUS_DONE_BIT) & 0x1u) == 0u);
    if ((status >> STATUS_ERROR_BIT) & 0x1u) {
        xil_printf("%s: the cascade controller reports an error, status 0x%08x\r\n", path_name[path], (unsigned int)status);
        return -1;
    }

    uint32_t result_register = ip_read(CASCADE_BASE_ADDR, REG_RESULT);
    uint32_t head_ran = (result_register >> RESULT_HEAD_RAN_BIT) & 0x1u;
    result->pred         = (uint8_t)(result_register & 0x3u);
    result->flags        = (uint8_t)(head_ran | (((result_register >> RESULT_EXCLUDED_SHIFT) & 0x3u) << 1));
    result->net_load_ns  = ip_read(CASCADE_BASE_ADDR, REG_NET_LOAD_CYCLES) * PL_NS_PER_CYCLE;
    result->net_run_ns   = ip_read(CASCADE_BASE_ADDR, REG_NET_RUN_CYCLES) * PL_NS_PER_CYCLE;
    result->head_load_ns = ip_read(CASCADE_BASE_ADDR, REG_HEAD_LOAD_CYCLES) * PL_NS_PER_CYCLE;
    result->head_run_ns  = ip_read(CASCADE_BASE_ADDR, REG_HEAD_RUN_CYCLES) * PL_NS_PER_CYCLE;
    result->total_ns     = ip_read(CASCADE_BASE_ADDR, REG_TOTAL_CYCLES) * PL_NS_PER_CYCLE;
    // The scores of the CNN, the SNN or (cascade) the selector are in the BRAM; the head wrote its score elsewhere.
    for (int c = 0; c < N_CLASSES; c++)
        result->scores[c] = score_q16_to_q8_8(bram_read_score(c));
    return 0;
}

// ============================================================
// MAIN
// ============================================================
int main(void)
{
    init_platform();
    usleep(1);                             // starts the global timer: XTime_GetTime() returns 0 until the first sleep call
    uart_set_baudrate(UART_BAUD);

    if (DFX_Init() != XST_SUCCESS) {
        cleanup_platform();
        return 1;
    }

    xil_printf("\r\nAssignment 2: UART image -> BRAM -> cascade controller (CNN, SNN, cascade)\r\n");
    xil_printf("UART_BAUD=%u  BRAM at 0x%08x  cascade controller at 0x%08x\r\n", (unsigned int)UART_BAUD, (unsigned int)BRAM_BASE_ADDR, (unsigned int)CASCADE_BASE_ADDR);
    xil_printf("RX_BYTES=%d\r\n", RX_BYTES);
    xil_printf("Reply: 0xA5 <uart_rx_us> <3 x (pred flags net_load_ns net_run_ns head_load_ns head_run_ns total_ns scores)> 0x5A\r\n");

    while (1)
    {
        uint32_t uart_rx_us;
        if (uart_receive_image(rx_buf, &uart_rx_us) != 0)
            continue;
        bram_store_image(rx_buf);

        PathResult result[N_PATHS] = {{0}};
        for (int p = 0; p < N_PATHS; p++)
            if (run_path((Path)p, &result[p]) != 0)
                result[p].pred = PRED_FAILED;

        uart_send_reply_frame(uart_rx_us, result);

        // Text goes out AFTER the reply frame (the PC discards anything between frames).
        for (int p = 0; p < N_PATHS; p++) {
            if (result[p].pred == PRED_FAILED)
                xil_printf("%s: FAILED\r\n", path_name[p]);
            else
                xil_printf("%s: %s, head %s, run %u+%u ns, load %u+%u ns, total %u ns, scores [%d %d %d]\r\n", path_name[p],
                           class_name[result[p].pred], (result[p].flags & 1) ? class_name[(result[p].flags >> 1) & 3] : "-",
                           (unsigned int)result[p].net_run_ns, (unsigned int)result[p].head_run_ns,
                           (unsigned int)result[p].net_load_ns, (unsigned int)result[p].head_load_ns,
                           (unsigned int)result[p].total_ns,
                           result[p].scores[0], result[p].scores[1], result[p].scores[2]);
        }
    }

    cleanup_platform();
    return 0;
}
