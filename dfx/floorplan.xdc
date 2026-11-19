# Floorplan of the two reconfigurable partitions of the xc7z020 (explicit site ranges: CLOCKREGION ranges are not available on 7 series).
# Placed sizes of the largest module in each slot: SNN (network_slot) 5.5k LUT, 2.5 BRAM tiles, 12 DSP48; head (head_slot) 2.9k LUT, 3 BRAM tiles, 7 DSP48.
# Pblock heights follow clock-region boundaries (50 slice rows), which RESET_AFTER_RECONFIG on 7 series requires.
# The configuration logic (ICAP sites) sits in the middle clock region X1Y1, so both partitions stay out of it: the static ICAPE2
# of the DFX Controller must be placed there.

# network_slot: 1600 slices, 6400 LUT, 20 RAMB36, 40 DSP48
create_pblock pblock_network_slot
add_cells_to_pblock [get_pblocks pblock_network_slot] [get_cells -quiet [list nn_dfx_design_i/network_slot]]
resize_pblock [get_pblocks pblock_network_slot] -add {SLICE_X16Y0:SLICE_X49Y49 \
                                                      DSP48_X1Y0:DSP48_X2Y19 \
                                                      RAMB18_X1Y0:RAMB18_X2Y19 \
                                                      RAMB36_X1Y0:RAMB36_X2Y9 \
}
set_property RESET_AFTER_RECONFIG true [get_pblocks pblock_network_slot]
set_property SNAPPING_MODE ON [get_pblocks pblock_network_slot]

# head_slot: 900 slices, 3600 LUT, 10 RAMB36, 20 DSP48
create_pblock pblock_head_slot
add_cells_to_pblock [get_pblocks pblock_head_slot] [get_cells -quiet [list nn_dfx_design_i/head_slot]]
resize_pblock [get_pblocks pblock_head_slot] -add {SLICE_X74Y0:SLICE_X95Y49 \
                                                   DSP48_X3Y0:DSP48_X3Y19 \
                                                   RAMB18_X4Y0:RAMB18_X4Y19 \
                                                   RAMB36_X4Y0:RAMB36_X4Y9 \
}
set_property RESET_AFTER_RECONFIG true [get_pblocks pblock_head_slot]
set_property SNAPPING_MODE ON [get_pblocks pblock_head_slot]

# Compressed bitstreams: the partial bitstreams are smaller and load faster through the ICAP.
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
